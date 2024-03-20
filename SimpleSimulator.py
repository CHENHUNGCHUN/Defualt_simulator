import pandas as pd
from scipy.stats import bernoulli
import numpy as np
import math


#年金function

#年金function
def loanpayment(amount,periods,rate,time,default_time=0):   #time 為了讓columns不要重複 然後第幾期投 #default_time 如果是6,代表第5期繳完以後就沒繳了
    d = amount / ((1-pow((1+(rate/1200)),(-periods)))/ (rate/1200))
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

###################################################################################################################

def main(amount,percent,default_rate,tatol_case,periods,rate,re_inve_time,fee,inter_fee,contract_periods):
    #先確定全部案件有哪幾筆是default
    _ = bernoulli.rvs(default_rate, size=tatol_case) #暫時先用白努力
    default_case = np.where(_ > 0)[0]

    #決定default的期數
    case_status = {} #key:案件 ,values:違約(呆帳)期數
    for i in range(tatol_case):
        if i not in default_case:
            case_status[i]=0
        else:
            _ = int(np.random.uniform(1,periods))#先暫時用uniform選default期數,先假設都是在一半以前就default
            case_status[i] = _

    #確認每輪要投多少件
    #先扣手續費
    real_amount = amount - (amount/(1+fee))*fee  #扣掉手續費後實際可以投資的額度
    inv_every_case_o = real_amount * percent #每件案件投資多少錢(固定)
    a = inv_every_case_o
    num_inv_case = math.ceil(real_amount/inv_every_case_o) #總共投多少案件

    #確認投的案件
    case=[]
    case_summary = [] #已經投過的案件集
    for i in range(int(num_inv_case)):
        _ = int(np.random.uniform(0,tatol_case))
        case.append(_)
        case_summary.append(_)

    #製表
    main_table = pd.DataFrame(columns=['total_amount','draw_amount','total_gain','sum_remain_principal','sum_reprin','sum_reint','re_invest','non-reinvest'])
    for k in range(re_inve_time):
            if k == 0:
                #第一次投入
                inv_every_case_o = a
                for i,j in zip(range(len(case)),case):
                    if i == len(case) - 1 : #最後一期用剩餘資金投入
                        inv_every_case_o = real_amount
                    else:
                        inv_every_case_o
                    main_table = pd.concat([main_table,loanpayment(inv_every_case_o,periods,rate,0,default_time=case_status[j])],axis=1)
                    real_amount = real_amount - inv_every_case_o

                main_table.loc[0,'total_gain'] = 0
                main_table.loc[0,'sum_remain_principal'] = sum(main_table.iloc[0,main_table.columns.str.startswith('remain_principal_')])
                main_table.loc[0,'sum_reprin'] = 0
                main_table.loc[0,'sum_reint'] = 0
                main_table.loc[0,'re_invest'] = 0
                main_table.loc[0,'non-reinvest'] = 0
            else:
                #先抓當期可以回收的金額 =>期初收回資金
                sum_reprin = sum(main_table.iloc[k,main_table.columns.str.startswith('re_principal_')])
                sum_reint = sum(main_table.iloc[k,main_table.columns.str.startswith('re_interest_')])*(1-inter_fee)
                main_table.loc[k,'sum_reprin'] = sum_reprin
                main_table.loc[k,'sum_reint'] = sum_reint
                # print(sum_reint,sum_reprin)

                #當期投資金額 => 期末投資
                inv_every_case_o = a  #還原inv_every_case_o
                total_gain = sum_reprin + sum_reint
                main_table.loc[k,'total_gain'] = total_gain
                main_table.loc[k,'draw_amount'] = 0  #暫時還沒開發到
                main_table.loc[k,'re_invest'] = (main_table.loc[k,'total_gain']- main_table.loc[k,'draw_amount'])
                main_table.loc[k,'non-reinvest'] = main_table.loc[k-1,'non-reinvest'] + main_table.loc[k,'total_gain'] - main_table.loc[k,'re_invest']

                real_amount = main_table.loc[k,'re_invest']
                real_amount = real_amount - (real_amount/(1+fee))*fee
                num_inv_case = math.ceil(real_amount/inv_every_case_o)
                #決定要投哪些case
                case=[]
                for i in range(int(num_inv_case)):
                    _ = int(np.random.uniform(0,tatol_case))
                    while _  in case_summary:
                        _ = int(np.random.uniform(0,tatol_case))
                    case.append(_)
                    case_summary.append(_)
                    #print(case_summary)
                #把年金表併入main_table
                for i,j in zip(range(len(case)),case):
                    if i == len(case) - 1 :
                        inv_every_case_o = real_amount
                    else:
                        inv_every_case_o
                    main_table = pd.concat([main_table,loanpayment(inv_every_case_o,periods,rate,k,default_time=case_status[j])],axis=1)
                    real_amount = real_amount-inv_every_case_o
                sum_remain_principal = sum(main_table.iloc[k,main_table.columns.str.startswith('remain_principal_')])
                main_table.loc[k,'sum_remain_principal'] = sum_remain_principal

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
        main_table.loc[j,'non-reinvest'] = main_table.loc[j-1,'non-reinvest'] + main_table.loc[j,'total_gain'] - main_table.loc[j,'re_invest'] 
    return main_table
###############################################################################################################################################

#################################################################################################################################################
#條件
percent = 0.01  #代操每筆投入只能占amount的百分比
default_rate = 0.0487 #件數的違約率
tatol_case = 1200 #先設定總案件量

#投資標的條件
amount = 1000000 #投資金額
periods = 24 #標的期數
rate = 14 #標的利率
re_inve_time = 24 #再投資期數  如果是24 代表只投資到23期期末

fee = 0.03  #手續費
inter_fee =0.3   #金流維護費

contract_periods = 24 #合約總期數
#################################################################################################################################################

table = main(amount,percent,default_rate,tatol_case,periods,rate,re_inve_time,fee,inter_fee,contract_periods)

#計算全部defualt金額
defualt =sum(table.iloc[:,table.columns.str.startswith('default_')].max())
defualt_time = len(np.where((table.iloc[:,table.columns.str.startswith('default_')].max()) > 0 )[0])

#總投資件數
invest_case = len(table.iloc[1,table.columns.str.startswith('default_')])

#計算合約到期平轉後總金額
total_gain = sum(table.loc[contract_periods,['sum_remain_principal','non-reinvest']]) #假設合約到期期末不投入,所以直接把"未投入金額" 跟 "目前已投入本金" 相加
total_gain_last_time = table.loc[len(table)-1,'non-reinvest']
print('總投資金額:',amount)
print('總投資件數:',invest_case)
print('平轉總金額(含利息):',total_gain)
print('持有到最後一期結束:',total_gain_last_time )
print('總違約件數:', defualt_time)
print('總違約金額:',defualt)
print('平轉總金額(含利息),加回呆帳金額:',total_gain+defualt)
print('持有到最後一期結束,加回呆帳金額:',total_gain_last_time+defualt )



table.to_csv('table.csv')
