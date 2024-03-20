from scipy.stats import bernoulli
from scipy.stats import beta
from multiprocessing import Process,Manager 
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

#################################################################################################################################################

#貝氏呆帳率的分布 (視覺化、1~4階動差)
def default_distribution(n,y):

    a0 = 0.5 #alpha 先驗
    b0 = 0.5 #beta 先驗
    #貝氏共軛
    a = a0 + y #alpha 後驗
    b = b0 + n -y  #beta 後驗

    x = np.linspace(0,1,1002)[1:-1]
    dist = beta(a=a,b=b)
    y = dist.pdf(x)
    plt.plot(x,y)
    plt.title('default distribution')
    plt.show()

    print(f'貝氏推論呆帳率95%的信賴區間{round(beta.ppf(0.025,60.5,1304.5),5)}~{round(beta.ppf(0.975,60.5,1304.5),5)}')
    mean, var, skew, kurt = beta.stats(60.5,1304.5,moments='mvsk')
    print('均值:', mean.astype('float16'), '變異數:', var.astype('float16'), '偏態:', skew.astype('float16'), '峰值:', kurt.astype('float16'))

#################################################################################################################################################

#頻率統計的呆帳率信賴區間
def freq_ci(n,y):
    p_hat = y/n
    q_hat = 1-p_hat
    upper_bound = p_hat + 1.96*(np.sqrt((p_hat*q_hat)/n))
    lower_bound = p_hat - 1.96*(np.sqrt((p_hat*q_hat)/n))
    print(f'頻率統計的呆帳率95%的信賴區間:{round(lower_bound,5)}~{round(upper_bound,5)}')

#################################################################################################################################################

#return 後驗的alpha跟beta
def default_r(n, y):
    a0 = 0.5 #alpha 先驗
    b0 = 0.5 #beta 先驗
    #貝氏共軛
    a = a0 + y #alpha 後驗
    b = b0 + n - y  #beta 後驗
    return a, b

#################################################################################################################################################

#年金function
def loanpayment(amount,periods,rate,time,default_time=0):   #time 為了讓columns不要重複 然後第幾期投 #default_time 如果是6,代表第5期繳完以後就沒繳了
    d = amount / ((1-pow((1+(rate/1200)),(-periods)))/ (rate/1200)) #年金公式
    df = pd.DataFrame(columns=[f'remain_principal_{time}',f're_principal_{time}',f're_interest_{time}',f'sum_re_{time}'])
    for i in range(0,periods+1+time):
        if i < time: 
            df.loc[i,f'remain_principal_{time}'] = 0
            df.loc[i,f're_principal_{time}'] = 0
            df.loc[i,f're_interest_{time}'] = 0
            df.loc[i,f'sum_re_{time}'] = 0
            df.loc[i,f'default_{time}'] = 0
        elif i == time:
            df.loc[i,f'remain_principal_{time}'] = amount
            df.loc[i,f're_principal_{time}'] = 0
            df.loc[i,f're_interest_{time}'] = 0
            df.loc[i,f'sum_re_{time}'] = 0
            df.loc[i,f'default_{time}'] = 0
        else:
            df.loc[i,f're_interest_{time}'] = df.loc[i-1,f'remain_principal_{time}'] * ((rate * 0.01) /12)
            df.loc[i,f're_principal_{time}'] = d - df.loc[i,f're_interest_{time}'] 
            df.loc[i,f'remain_principal_{time}'] = df.loc[i-1,f'remain_principal_{time}'] - df.loc[i,f're_principal_{time}']
            df.loc[i,f'sum_re_{time}'] = df.loc[i,f're_principal_{time}']+df.loc[i,f're_interest_{time}']
    if default_time == 0:
        return df
    else:
        df.loc[time+default_time:,[f'remain_principal_{time}',f're_interest_{time}',f're_principal_{time}',f'sum_re_{time}']]=0
        df.loc[time+default_time:,f'default_{time}'] = df.loc[default_time+time-1,f'remain_principal_{time}']
        df[f'default_{time}'].fillna(0,inplace=True)
        return df
    
##########################################################################################################################################

def main(amount,n,y,percent,tatol_case,periods,rate,re_inve_time,fee,inter_fee,contract_periods,contract_fee):
    #先確定全部案件有哪幾筆是default
    a,b=default_r(n,y)
    default_rate = beta.rvs(a,b,size=1)
    _ = bernoulli.rvs(default_rate, size=tatol_case) #暫時先用白努力
    default_case = np.where(_ > 0)[0]  #資料在0的位置

    #決定default的期數
    case_status = {} #key:案件 ,values:違約(呆帳)期數
    for i in range(tatol_case):   #0~tatol_case(不包含)
        if i not in default_case:
            case_status[i]=0
        else:
            _ = int(np.random.uniform(1,periods*0.7))#先暫時用uniform選default期數,先假設都是uniform 以後再改(0.7是因為目前有default的狀況,都是前70%的期數)
            case_status[i] = _

    inv_every_case_o = amount * percent #每件案件投資多少錢(固定)
    orang_amount = amount 
    a = inv_every_case_o
    case_summary = [] #已經投過的案件集

    contract_fee_list = [11+i*12 for i in range(int(contract_periods/12)-1)] #收契約費用的期數 目前只支援可以被12整除的契約

    #製表
    main_table = pd.DataFrame(columns=['total_amount','draw_amount',
                                       'total_gain','sum_remain_principal','sum_reprin','sum_reint',
                                       're_invest',
                                       'total_fee','inter_fee_money',
                                       'non-reinvest'])
    for k in range(re_inve_time):
            if k == 0:
                #第一次投入
                inv_every_case_o = a #初始化固定每筆案件投資金額
                total_fee = 0 #初始化總手續費
                amount =  amount #本次可以投入的總額
                #第一筆要先扣掉amount * contract_fee
                amount = orang_amount*(1-contract_fee) 

                for i in range(int(amount/inv_every_case_o)+5):  
                    if amount < 1 : #最後一期用剩餘資金投入
                        break
                    elif amount < inv_every_case_o:
                        inv_every_case_o = amount
                    else:
                        inv_every_case_o = inv_every_case_o
                    #投入資金 = 投入資金 - 手續費
                    invest_amount = inv_every_case_o / (1+fee)
                    total_fee += invest_amount * fee
                    #隨機選一個案子投 然後從案件diction刪除
                    j = random.choice(list(case_status.keys()))
                    #目前已投的案子
                    case_summary.append(j)
                    main_table = pd.concat([main_table,loanpayment(invest_amount,periods,rate,0,default_time=case_status[j])],axis=1)
                    del case_status[j]
                    amount = amount - invest_amount-invest_amount * fee
                
                main_table.loc[0,'total_amount'] = orang_amount
                main_table.loc[0,'total_gain'] = 0
                main_table.loc[0,'sum_remain_principal'] = sum(main_table.iloc[0,main_table.columns.str.startswith('remain_principal_')])
                main_table.loc[0,'sum_reprin'] = 0
                main_table.loc[0,'sum_reint'] = 0
                main_table.loc[0,'re_invest'] = 0
                main_table.loc[0,'non-reinvest'] = 0
                main_table.loc[0,'total_fee'] = total_fee
                main_table.loc[0,'inter_fee_money'] = 0

            else:
                #先抓當期可以回收的金額 =>期初收回資金
                sum_reprin = sum(main_table.iloc[k,main_table.columns.str.startswith('re_principal_')])
                sum_reint = sum(main_table.iloc[k,main_table.columns.str.startswith('re_interest_')])*(1-inter_fee)
                main_table.loc[k,'inter_fee_money'] = sum(main_table.iloc[k,main_table.columns.str.startswith('re_interest_')])*(inter_fee) #金流管理費
                main_table.loc[k,'sum_reprin'] = sum_reprin
                main_table.loc[k,'sum_reint'] = sum_reint

                #當期投資金額 => 期末投資
                main_table.loc[k,'total_gain'] = sum_reprin + sum_reint
                if k in contract_fee_list:
                    main_table.loc[k,'draw_amount'] = orang_amount *(contract_fee)   
                else:
                    main_table.loc[k,'draw_amount'] = 0
                #可再投資金額 =>本期總收益(total_gain) - 提領金額
                main_table.loc[k,'re_invest'] = (main_table.loc[k,'total_gain']- main_table.loc[k,'draw_amount'])
                #本期未投資金額 =>上一期的未投入金額 + 本期總收益(total_gain) - 再投入金額 - 抽取的資金(他是出金 不是留在帳戶內)
                main_table.loc[k,'non-reinvest'] = main_table.loc[k-1,'non-reinvest'] + main_table.loc[k,'total_gain'] - main_table.loc[k,'re_invest'] - main_table.loc[k,'draw_amount']

                #先確定可以投資多少金額
                amount = main_table.loc[k,'re_invest'] #本次可以投入的總額
                inv_every_case_o = a  #初始化固定每筆案件投資金額
                total_fee = 0 #初始化總手續費
                for i in range(int(amount/inv_every_case_o)+5):
                    if amount < 1 : #最後一期用剩餘資金投入
                        break
                    elif amount < inv_every_case_o:
                        inv_every_case_o = amount
                    else:
                        inv_every_case_o = inv_every_case_o
                    #投入資金 = 投入資金 - 手續費
                    invest_amount = inv_every_case_o / (1+fee)
                    total_fee += invest_amount * fee
                    #隨機選一個案子投:
                    j = random.choice(list(case_status.keys()))
                    #目前已投的案子
                    case_summary.append(j)
                    main_table = pd.concat([main_table,loanpayment(invest_amount,periods,rate,k,default_time=case_status[j])],axis=1)
                    del case_status[j]
                    amount = amount - invest_amount-invest_amount * fee
                
                sum_remain_principal = sum(main_table.iloc[k,main_table.columns.str.startswith('remain_principal_')])
                main_table.loc[k,'sum_remain_principal'] = sum_remain_principal
                main_table.loc[k,'total_fee'] = total_fee
                main_table = main_table.fillna(0)

    main_table = main_table.fillna(0)

    #計算不再投入後的年金表
    for j in range(re_inve_time,len(main_table)):
        sum_reprin = sum(main_table.iloc[j,main_table.columns.str.startswith('re_principal_')])
        sum_reint = sum(main_table.iloc[j,main_table.columns.str.startswith('re_interest_')]) *(1-inter_fee) 
        sum_remain_principal = sum(main_table.iloc[j,main_table.columns.str.startswith('remain_principal_')])
        main_table.loc[j,'sum_reprin'] = sum_reprin
        main_table.loc[j,'sum_reint'] = sum_reint
        main_table.loc[j,'total_gain'] = main_table.loc[j,'sum_reint']+main_table.loc[j,'sum_reprin']
        main_table.loc[j,'sum_remain_principal'] = sum_remain_principal
        main_table.loc[j,'draw_amount'] = 0  #暫時還沒開發到
        main_table.loc[j,'re_invest'] = 0
        main_table.loc[j,'total_fee'] = 0 #因為沒有再投入了 就不會再收手續費
        main_table.loc[j,'non-reinvest'] = main_table.loc[j-1,'non-reinvest'] + main_table.loc[j,'total_gain'] - main_table.loc[j,'re_invest'] 
        main_table.loc[j,'inter_fee_money'] = sum(main_table.iloc[j,main_table.columns.str.startswith('re_interest_')]) *(inter_fee) 
    return main_table

###############################################################################################################################################

def metrics(table):
    defualt =sum(table.iloc[:,table.columns.str.startswith('default_')].max())   #總違約金額
    defualt_time = len(np.where((table.iloc[:,table.columns.str.startswith('default_')].max()) > 0 )[0])  #總違約件數
    invest_case = len(table.iloc[1,table.columns.str.startswith('default_')])  #總投資件數
    risk_insure =sum(table.loc[:,'total_fee'])*((fee-0.02)/(fee))  #總風保費  0.02代表交易的手續費
    total_fee =sum(table.loc[:,'total_fee'])*0.01/fee #總手續費
    contrac_total_gain = sum(table.loc[contract_periods,['sum_remain_principal','non-reinvest']])  #合約到期後直接平轉總收益
    contrac_gain_last_time = table.loc[len(table)-1,'non-reinvest']  #放到最後一個投資件數到期
    return [defualt,defualt_time,invest_case,risk_insure,contrac_total_gain,contrac_gain_last_time,total_fee]

###############################################################################################################################################

def final(amount,n,y,percent,tatol_case,periods,rate,re_inve_time,fee,inter_fee,contract_periods,contract_fee,metrics_list,run_time):
    for i in tqdm(range(run_time)):
        table = main(amount,n,y,percent,tatol_case,periods,rate,re_inve_time,fee,inter_fee,contract_periods,contract_fee)
        metri = metrics(table)
        metrics_list.append(metri)

###############################################################################################################################################
#測試table內容有沒有錯誤用
def final_2(amount,n,y,percent,tatol_case,periods,rate,re_inve_time,fee,inter_fee,contract_periods,contract_fee,metrics_list,run_time):
    table = main(amount,n,y,percent,tatol_case,periods,rate,re_inve_time,fee,inter_fee,contract_periods,contract_fee)
    return table
###############################################################條件###############################################################################
#呆帳資訊 (主要去更新 n,y ,其他東西default_r 跟default_distribution 都已經寫好)
n = 555 #總案件數    389 (1對1 166件) =>共555件
y = 53  #default案件數   36 (1對1 17件) =>44件


#代操條件
percent = 0.01  #代操每筆投入只能占amount的百分比   0.01
tatol_case = 1200 #總案件量(直接用呆帳資訊的n)  注意投資期數太多  總案件量 要注意夠不夠 ,但總案件量跟總投資案比率不能差太多 不然會低估呆帳金額


#投資標的條件
amount = 1000000 #投資金額
rate = 15 #標的利率
periods = 24 #標的期數
re_inve_time = 36 #再投資期數  如果是24 代表只投資到23期期末   注意投資期數太多  總案件量 要注意夠不夠
contract_periods = 36 #合約總期數 目前版本一定要可以被12整除 注意總案件量夠不夠


#費用
fee = 0.095  #手續費  如果是0的話metrics會有問題   0.02 =>交易手續費   fee-0.02=>風保費(調整風保費在第225行)
inter_fee = 0.2  #金流維護費
contract_fee = 0.01 #期初扣的費用


#程序
run_time = 30  #function總共要跑幾次 ,看要用多少個core_number  * run_time 就可以得到多少次
core_number = os.cpu_count()-1  #調整要用多少個core去跑
#################################################################################################################################################

'''
metrics
'總投資件數:',metri[2]
'平轉總金額(含利息):',metri[-3]
'持有到最後一期結束:',metri[-2]
'平轉總金額(含利息),加回呆帳金額及報酬率:',metri[-3]+metri[0], ((metri[-3]+metri[0])/amount)-1
'持有到最後一期結束,加回呆帳金額及報酬率:',metri[-2]+metri[0], ((metri[-2]+metri[0])/amount)-1
'總呆帳件數:', metri[1]
'總呆帳金額:',metri[0]
'總手續費:',metri[-1]
'''
##############################################################################################################################################
#for test
# metrics_list = []
# table = final_2(amount,n,y,percent,tatol_case,periods,rate,re_inve_time,fee,inter_fee,contract_periods,contract_fee,metrics_list,run_time)
# print(table.loc[contract_periods,['sum_remain_principal','non-reinvest']])
# table.to_excel('test_20231003.xlsx',index=False)
##############################################################################################################################################


if __name__ == '__main__':
    with Manager() as manager:
        #manager
        metrics_list = manager.list()

        #spawn
        processes = []
        for i in range(core_number):
            processes.append(Process(target=final ,args=(amount,n,y,percent,tatol_case,
                                                         periods,rate,re_inve_time,fee,inter_fee,
                                                         contract_periods,contract_fee,metrics_list,run_time)
                                                         ))
        
        start = time.time() #計時器開始
        for process in processes:
            process.start()    
        for process in processes:
            process.join()
        end = time.time() #計時器結束

        #print(metrics_list)
        print('總執行次數:',len(metrics_list),'次')
        print("執行時間：%f 秒" % (end - start))
        default_distribution(n,y)
        freq_ci(n,y)

        defualt_money = []
        risk_insure_money = []
        customer_return = []
        invest_case_list = []
        defualt_time_list =[]

        for i in range(len(metrics_list)):
            _ = metrics_list[i][0]  #呆帳金額
            _a = metrics_list[i][3] 
            _b = metrics_list[i][-3] + metrics_list[i][0]
            _c = metrics_list[i][2]
            _d = metrics_list[i][1]

            defualt_money.append(_)
            risk_insure_money.append(_a)
            customer_return.append(_b)
            invest_case_list.append(_c)
            defualt_time_list.append(_d)
        
        print('平均總投資件數',(sum(invest_case_list)/len(invest_case_list)))
        print('平均總呆帳件數',(sum(defualt_time_list)/len(defualt_time_list)))
        print('平均呆帳金額:',sum(defualt_money)/len(defualt_money))
        print('平均風保費金額:',sum(risk_insure_money)/len(risk_insure_money))
        print('客戶平均期末平轉拿回金額',sum(customer_return)/len(customer_return))
        print('客戶平均期末平轉報酬率',((sum(customer_return)/len(customer_return))/amount)-1)
        








