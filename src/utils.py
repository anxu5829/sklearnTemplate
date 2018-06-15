import numpy as np
import pandas as pd
import datetime as datetime
from pyspark.sql.functions import udf
import logging
import  gc
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import time
from  scipy.sparse import csc_matrix
import pickle
from operator import itemgetter
from collections import Counter


def save_pickle(Object, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(Object, outfile, pickle.HIGHEST_PROTOCOL)
def load_pickle(filename):
    with open(filename, 'rb') as infile:
        Object = pickle.load(infile)
    return Object



def findExtraCols(func):
        def wrapper(self, *args, **kwargs):
            currentCols = set(self.data.columns.tolist())
            #print(currentCols)
            typeOfCols = func(self,*args, **kwargs)
            colsAfterDealing = set(self.data.columns.tolist())
            if typeOfCols == 'continue':
                self.newContinueCols.extend(list(colsAfterDealing - currentCols))
            if typeOfCols == 'category':
                self.newCategoryCols.extend(list(colsAfterDealing - currentCols))
        return wrapper

class DataPreprocessingModule(object):
    def __init__(self,data):
        self.data = data
        self.newCategoryCols = []
        self.newContinueCols = []

    @findExtraCols
    def _timePreprocessing(self,timeStampCol):
        self.data['time'] = pd.to_datetime(self.data[timeStampCol],unit= 's')
        # data['time'] = data['time'].apply(lambda x: x + datetime.timedelta(hours=8))
        self.data['day'] = self.data['time'].apply(lambda x: int(str(x)[8:10]))
        self.data['hour'] = self.data['time'].apply(lambda x: int(str(x)[11:13])) 
        self.data['minute'] = self.data['time'].apply(lambda x: int(str(x)[14:16]))

        # mapping hour and minutes
        self.data['maphour'] = pd.cut(self.data['hour'],[0,6,12,18,24]).cat.codes
        self.data['mapmin'] = self.data['minute'] % 15 + 1

        # time is not useful aborted
        self.data.pop('time')


        return "continue"
    
    @findExtraCols
    def _categoryPreprocessing(self,categoryName,frac = 0.95):
        
        # get 二级标题,三级标题
        self.data.loc[:,'secondLevelCatagory'] = self.data[categoryName].str.split(';').map(lambda x: x[1])
        valuecounts = self.data['secondLevelCatagory'].value_counts()
        valuecounts = valuecounts[valuecounts.cumsum()<self.data.shape[0]*frac ]
        secondTopN  = valuecounts.to_dict()
        sorting  = (sorted(secondTopN.items(),key = itemgetter(1),reverse = False))
        sorting  = { i[0]:(idx+1) for idx , i in enumerate(sorting)} 
        sorting  = self.data['secondLevelCatagory'].map(sorting)
        self.data['secondLevelCatagory_Most_Freq'] = sorting.fillna(0)
        self.data.pop('secondLevelCatagory')
        del valuecounts
        del sorting
        gc.collect()
        return "continue"



    @findExtraCols
    def _propertyPreprocessing(self,propertyName , frac = 0.8,minCount  = 40):
        
        temp = self.data[propertyName].str.split(";").map(lambda x: [ int(i) for i in x])
        propertyList = pd.Series( np.concatenate(temp))
        valuecounts  = propertyList.value_counts().cumsum()
        valuecounts = valuecounts[valuecounts <  valuecounts.values[-1] *frac]
        if valuecounts.shape[0] > 50:
            valuecounts = valuecounts.iloc[:minCount]
        topN  = valuecounts.index.values
        for i in topN:
            self.data.loc[:,'preperty_'+str(i)] = propertyList.map(lambda x: i in temp)
            logging.info("property " + str(i) + "Done.")
        del propertyList
        del valuecounts
        del topN
        gc.collect()
        
        return "continue"
        

    @findExtraCols
    def _predictCategory(self,predictCategoryName):

        import re
        catPatten = '\d+(?=:)'
        propertyPatten = '(?<=:)\w+(?=;)'
        predictCat = self.data[predictCategoryName].map(
            lambda s : set(
                        [ int(i) for i in re.findall(catPatten,s) ]
                        )
            )
        predictProperty = self.data[predictCategoryName].map(
            lambda s : set(
                        [int(i) for i in re.findall(propertyPatten,s) ]
                        )
            )

        category  = self.data['item_category_list'].str.split(";").map(lambda x : set([int(i) for i in x]))
        propertys = self.data['item_property_list'].str.split(";").map(lambda x: set([int(i) for i in x]))

        self.data.loc[:,'hitCategory'] = (category - predictCat).map(lambda x: len(x) ).fillna(-1)
        self.data.loc[:,'hitProperty'] = (propertys - predictProperty).map(lambda x: len(x)).fillna(-1)

        del category,propertys
        del predictCat,predictProperty
        gc.collect()
        return "continue"



    def _labelEncoder(self,col_need_encode):
        for col in col_need_encode:
            col_encoder = LabelEncoder()
            col_encoder.fit(self.data[col]) 
            self.data[col] = col_encoder.transform(self.data[col])

class FeatureEngineeringModule(object):
    def __init__(self,data):
        self.data = data
        self.newCategoryCols = []
        self.newContinueCols = []

    @findExtraCols
    def _genExposeRatioThroughSth(self,genList):
        # 这里考虑的是这么一个问题：
        # 1 比如item_id ，在每一个小时里头，产生了多少次浏览记录，和占总浏览记录的比值
        # _exposeRatioThroughSth(data, “user_id”,"hour") 表明统计用户在每个时间的点击次数占比

        def _exposeRatioThroughSth(ids,sthGrain):
            
            exp_sth = self.data.groupby([ids,sthGrain]).size().to_frame()
            cnt_name = ids+'_'+sthGrain+'_'+'cnt'
            ratio_name = ids+'_'+sthGrain+'_'+'ratio'
            exp_sth.columns = [cnt_name]
            exp_sth = exp_sth.reset_index()
            exp_sth[ratio_name] = round(
                exp_sth[cnt_name] / exp_sth.groupby(ids)[cnt_name].transform(np.sum)
                ,4)
            self.data = self.data.merge(exp_sth,how = 'left',on = [ids,sthGrain])

        for ids,sthGrain in genList:
            _exposeRatioThroughSth(ids,sthGrain)
        return "continue"

    @findExtraCols
    def _genAggregateLevelThroughSth(self,genDict,aggs =[ "mean"]):
        # 查看变量在另一个变量上的平均水平
        # 如_averageLevelThroughSth(data,"item_id","hour")
        # 表示统计每一个item_id的平均购买时间
        def _aggregateLevelThroughSth(ids,sthGrain,agg):
            grouped = self.data.groupby(ids).agg({sthGrain:agg}).reset_index()
            grouped.columns = [ids,ids+'_TO_'+sthGrain+'_'+agg]
            self.data = self.data.merge(grouped,how = 'left', on = ids)
        for sth,idList in genDict.items():
            for ids in idList:
                for agg in aggs:
                    _aggregateLevelThroughSth(ids,sth,agg)
        return "continue"
    @findExtraCols
    def _genActivePerTimeGrain(self,genActiveDict):
        # 对应第二行：统计item_category_list中，每天都有多少item_id的记录和user_id的记录
        #    {
        #             "day":{
        #                 "item_category_list":["item_id","user_id"]
        #                 "user_id":["item_city_id","shop_id","item_brand_id"],
        #                 "item_id":["user_id",""],
        #                 "shop_id":["user_id"],

        #                 "item_brand_id":["user_id"]
        #             }
        #         }

        # _activePerTimeGrain("item_category_list","item_id","day")
        # 表明统计每天，每一个item_category_list下有多少个item_id有记录
        def _activePerTimeGrain(ids1,ids2,sthGrain):
            add = pd.DataFrame(
                self.data.groupby([ids1,sthGrain])[ids2].unique().map(lambda x: len(x))
                ).reset_index()
            add.columns = [ids1,sthGrain,ids1+'_'+sthGrain+"_"+"_active_"+ids2+'__num']
            self.data = self.data.merge(add,on = [ids1,sthGrain],how = 'left')
        
        for sthGrain in genActiveDict.keys():
            for ids1,ids2List in genActiveDict[sthGrain].items():
                for ids2 in ids2List:
                    _activePerTimeGrain(ids1,ids2,sthGrain)
        return "continue"
    
    @findExtraCols
    def _genMaxBofA(self,genMaxBofADict):
        for A in genMaxBofADict.keys():
            for B in  genMaxBofADict[A]:
                
                add = self.data.groupby([A,B]).size().to_frame().reset_index().rename(columns = {B:"B",0:"BofA"})
                add = add.groupby(A).apply(lambda df : df.nlargest(1,"BofA").loc[:,"B"]).reset_index().loc[:,[A,'B']].rename(columns = {"B":B})
                self.data = self.data.merge(add,on = [A,B],how = 'left')
        return "continue"


    @findExtraCols
    def _genRelativeRatioToAvg(self,genRelativeRatioToAvgDict):

        # 第一行第一个：统计每个item_brand_id 的 price_level的均值
        # 并对与每一条观测的商品，查看其售价level和商铺的均值的比例（相对于商铺的其他商品的贵/便宜）
        #    {
        #         "item_brand_id":["item_price_level","item_sales_level","item_collected_level"]
        #         ,"item_city_id":["item_price_level","item_sales_level","item_collected_level"]
        #         ,"item_category_list":["item_price_level","item_sales_level","item_collected_level"]
        #     }
        for ids in genRelativeRatioToAvgDict:
            ids2 = genRelativeRatioToAvgDict[ids]
            df_g = self.data.groupby(ids)[ids2].mean().reset_index()
            colnames = [i for i in df_g.columns]
            for i in range(len(colnames)):
                if colnames[i] != ids:
                    colnames[i] += '_mean_by_' + ids
            df_g.columns = colnames
            self.data = pd.merge(self.data,df_g,how = 'left',on = ids)
            colnames = colnames[1:]
            for  i in range(len(colnames)):
                self.data[colnames[i]+'ratio'] = round(
                    (self.data[ids2[i]]/self.data[colnames[i]])
                    ,5
                )
        return "continue"

    # add some special tricks dealing with data ,
    # this will be change when facing other problem
    @findExtraCols
    def doTricks1(self):
        self.data.sort_values(['user_id','context_timestamp'],inplace =True)
        def _checkFirstAndLast(subset):
            subset2 = subset.copy()
            subset2.remove('day')
            str_used = 'click_'
            for i in subset2:
                str_used += i + '_'
            str_used += 'tab'
            self.data[str_used] = 0
            pos = self.data.duplicated(subset = subset , keep  = False)
            self.data.loc[pos, str_used] = 1
            pos = (~self.data.duplicated(subset=subset, keep='first')) & self.data.duplicated(subset=subset, keep=False)
            self.data.loc[pos, str_used] = 2
            pos = (~self.data.duplicated(subset=subset, keep='last')) & self.data.duplicated(subset=subset, keep=False)
            self.data.loc[pos, str_used] = 3
            del pos
            gc.collect()
        # 统计每天用户第一次点击和最后一次点击的时间戳
        subset = ['user_id','day']
        _checkFirstAndLast(subset)
        
        # 统计用户对于每一个商品在那一天的第一次点击和最后一次点击
        subset = ['item_id', 'user_id', 'day']
        _checkFirstAndLast(subset)

        # 统计用户对于品牌的第一次点击和最后一次点击
        subset = ['item_brand_id','user_id', 'day']
        _checkFirstAndLast(subset)

        subset = ['shop_id','user_id', 'day']
        _checkFirstAndLast(subset)

        subset = ['item_city_id','user_id', 'day']
        _checkFirstAndLast(subset)
        return "continue"

    @findExtraCols
    def doTricks2(self):
        self.data.sort_values(['user_id', 'context_timestamp'], inplace=True)
        #user_id

        # 首先，统计出每个用户每一天第一次点击的时间
        subset = ['user_id', 'day']
        temp = self.data.loc[:,['context_timestamp', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
        temp.rename(columns={'context_timestamp': 'u_day_diffTime_first'}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=subset)
        # 再统计出这个用户其余点击时间和第一个点击时间之间的差
        self.data['u_day_diffTime_first'] = self.data['context_timestamp'] - self.data['u_day_diffTime_first']
        del temp
        gc.collect()

        # 其次，统计出每一个用户每一天最后一次点击
        temp = self.data.loc[:,['context_timestamp', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
        temp.rename(columns={'context_timestamp': 'u_day_diffTime_last'}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=subset)
        # 再统计处这个用户其余时间和最后一次点击时间之间的差距
        self.data['u_day_diffTime_last'] = self.data['u_day_diffTime_last'] - self.data['context_timestamp']
        del temp
        gc.collect()
        # 若一个人一天只有一次点击，那么置这个变量的值为 1 
        self.data.loc[~self.data.duplicated(subset=subset, keep=False), ['u_day_diffTime_first', 'u_day_diffTime_last']] = -1
        
        #item_id
        subset = ['item_id', 'day']
        temp = self.data.loc[:,['context_timestamp', 'item_id', 'day']].drop_duplicates(subset=subset, keep='first')
        temp.rename(columns={'context_timestamp': 'i_day_diffTime_first'}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=subset)
        self.data['i_day_diffTime_first'] = self.data['context_timestamp'] - self.data['i_day_diffTime_first']
        del temp
        gc.collect()
        temp = self.data.loc[:,['context_timestamp', 'item_id', 'day']].drop_duplicates(subset=subset, keep='last')
        temp.rename(columns={'context_timestamp': 'i_day_diffTime_last'}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=subset)
        self.data['i_day_diffTime_last'] = self.data['i_day_diffTime_last'] - self.data['context_timestamp']
        del temp
        gc.collect()
        self.data.loc[~self.data.duplicated(subset=subset, keep=False), ['i_day_diffTime_first', 'i_day_diffTime_last']] = -1

        gc.collect()

        #item_brand_id, user_id
        subset = ['item_brand_id', 'user_id', 'day']
        temp = self.data.loc[:,['context_timestamp', 'item_brand_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
        temp.rename(columns={'context_timestamp': 'b_day_diffTime_first'}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=subset)
        self.data['b_day_diffTime_first'] = self.data['context_timestamp'] - self.data['b_day_diffTime_first']
        del temp
        gc.collect()
        temp = self.data.loc[:,['context_timestamp', 'item_brand_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
        temp.rename(columns={'context_timestamp': 'b_day_diffTime_last'}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=subset)
        self.data['b_day_diffTime_last'] = self.data['b_day_diffTime_last'] - self.data['context_timestamp']
        del temp
        gc.collect()
        self.data.loc[~self.data.duplicated(subset=subset, keep=False), ['b_day_diffTime_first', 'b_day_diffTime_last']] = -1


        #shop_id, user_id
        subset = ['shop_id', 'user_id', 'day']
        temp = self.data.loc[:,['context_timestamp', 'shop_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
        temp.rename(columns={'context_timestamp': 's_day_diffTime_first'}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=subset)
        self.data['s_day_diffTime_first'] = self.data['context_timestamp'] - self.data['s_day_diffTime_first']
        del temp
        gc.collect()
        temp = self.data.loc[:,['context_timestamp', 'shop_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
        temp.rename(columns={'context_timestamp': 's_day_diffTime_last'}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=subset)
        self.data['s_day_diffTime_last'] = self.data['s_day_diffTime_last'] - self.data['context_timestamp']
        del temp
        gc.collect()
        self.data.loc[~self.data.duplicated(subset=subset, keep=False), ['s_day_diffTime_first', 's_day_diffTime_last']] = -1
        return "continue"

    @findExtraCols
    def lasttimeDiff(self):
        self.data.sort_values(by = 'context_timestamp',inplace =True,ascending = True)
        for column in ['user_id', 'item_id']:
            gc.collect()
            self.data[column+'_lasttime_diff'] = 0
            # 拿到时间戳和其他信息
            train_data = self.data[['context_timestamp', column, column+'_lasttime_diff']].values
    
            lasttime_dict = {}
            for df_list in train_data:
                # 每次拿到一条记录(3,) ， （'context_timestamp', column, column+'_lasttime_diff'）
                if df_list[1] not in lasttime_dict:
                    # 判断对于这个user/item,是否存在上一个时刻的记录
                    df_list[2] = -1
                    # 将这一时刻记录下来
                    lasttime_dict[df_list[1]] = df_list[0]
                else:
                    # 记录和上一个观测记录之间的时间差
                    df_list[2] = df_list[0] - lasttime_dict[df_list[1]]
                    lasttime_dict[df_list[1]] = df_list[0]
            self.data[['context_timestamp', column, column+'_lasttime_diff']] = train_data
        return "continue"

    @findExtraCols
    def nexttimeDiff(self):
        self.data.sort_values(by = 'context_timestamp',inplace = True,ascending = False)
        for column in ['user_id', 'item_id']:
            gc.collect()
            self.data[column+'_nexttime_diff'] = 0
            train_data = self.data[['context_timestamp', column, column+'_nexttime_diff']].values
            nexttime_dict = {}
            for df_list in train_data:
                if df_list[1] not in nexttime_dict:
                    df_list[2] = -1
                    nexttime_dict[df_list[1]] = df_list[0]
                else:
                    df_list[2] = nexttime_dict[df_list[1]] - df_list[0]
                    nexttime_dict[df_list[1]] = df_list[0]
            self.data[['context_timestamp', column, column+'_nexttime_diff']] = train_data
        return "continue"


    @findExtraCols
    def dorollWin(self):

        self.data['context_timestamp_str'] = self.data['context_timestamp'].astype(str)
        user_time_join = self.data.groupby('user_id')['context_timestamp_str'].agg(lambda x:';'.join(x)).reset_index()
        user_time_join.rename(columns={'context_timestamp_str':'user_time_join'},inplace = True)
        self.data = pd.merge(self.data,user_time_join,on=['user_id'],how='left')
        user_shop_time_join = self.data.groupby(['user_id','shop_id'])['context_timestamp_str'].agg(lambda x:';'.join(x)).reset_index()
        user_shop_time_join.rename(columns={'context_timestamp_str':'user_shop_time_join'},inplace = True)
        self.data = pd.merge(self.data,user_shop_time_join,on=['user_id','shop_id'],how='left')
        user_item_time_join = self.data.groupby(['user_id','item_id'])['context_timestamp_str'].agg(lambda x:';'.join(x)).reset_index()
        user_item_time_join.rename(columns={'context_timestamp_str':'user_item_time_join'},inplace = True)
        self.data = pd.merge(self.data,user_item_time_join,on=['user_id','item_id'],how='left')
        self.data['index_']=self.data.index
        del user_time_join,user_shop_time_join,user_item_time_join
        
        nowtime=self.data.context_timestamp.values
        user_time=self.data.user_time_join.values
        user_shop_time=self.data.user_shop_time_join.values
        user_item_time=self.data.user_item_time_join.values
        
        data_len=self.data.shape[0]
        user_time_10_bf=np.zeros(data_len)
        user_time_10_af=np.zeros(data_len)
        user_shop_time_10_bf=np.zeros(data_len)
        user_shop_time_10_af=np.zeros(data_len)
        user_item_time_10_bf=np.zeros(data_len)
        user_item_time_10_af=np.zeros(data_len)


        # 逐行数据进行处理
        for i in range(data_len):
            # df1 : 拿到当前行的时间
            df1=nowtime[i]
            # 拿到user 所有经历的时间
            df2=user_time[i].split(';')
            df2_len=len(df2)
            for j in range(df2_len):
                # 查看用户购买的这些之间中，和这一次相近的（时间差小于600的次数）和与这一次不相近的次数
                if ((int(df2[j])-df1)<600) & ((int(df2[j])-df1)>0):
                    user_time_10_bf[i]+=1
                if ((int(df2[j])-df1)>-600) & ((int(df2[j])-df1)<0):
                    user_time_10_af[i]+=1
            
            # df3 ：用于对ueser_shop作这个操作

            df3=user_shop_time[i].split(';')
            df3_len=len(df3)
            for j in range(df3_len):
                if ((int(df3[j])-df1)<600) & ((int(df3[j])-df1)>0):
                    user_shop_time_10_bf[i]+=1
                if ((int(df3[j])-df1)>-600) & ((int(df3[j])-df1)<0):
                    user_shop_time_10_af[i]+=1
                    
            df4=user_item_time[i].split(';')
            df4_len=len(df4)
            for j in range(df4_len):
                if ((int(df4[j])-df1)<600) & ((int(df4[j])-df1)>0):
                    user_item_time_10_bf[i]+=1
                if ((int(df4[j])-df1)>-600) & ((int(df4[j])-df1)<0):
                    user_item_time_10_af[i]+=1
                    

        
        self.data['user_count_10_bf']=user_time_10_bf
        self.data['user_count_10_af']=user_time_10_af
        self.data['user_shop_count_10_bf']=user_shop_time_10_bf
        self.data['user_shop_count_10_af']=user_shop_time_10_af
        self.data['user_item_count_10_bf']=user_item_time_10_bf
        self.data['user_item_count_10_af']=user_item_time_10_af

        drops = ['context_timestamp_str', 'user_time_join', 'user_shop_time_join',
        'user_item_time_join', 'index_']
        self.data = self.data.drop(drops, axis=1)
        return "continue"

    @findExtraCols
    def doSize(self):

        add = pd.DataFrame(self.data.groupby(["shop_id", "day"]).item_id.nunique()).reset_index()
        add.columns = ["shop_id", "day", "shop_item_unique_day"]
        self.data = self.data.merge(add, on=["shop_id", "day"], how="left")

        user_query_day = self.data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_id_query_day'})
        self.data = pd.merge(self.data, user_query_day, how='left', on=['user_id', 'day'])
        
        self.data['min_10'] = self.data['minute'] // 10
        self.data['min_15'] = self.data['minute'] // 15
        self.data['min_30'] = self.data['minute'] // 30
        self.data['min_45'] = self.data['minute'] // 45
        
        # user 不同时间段点击次数
        min10_user_click = self.data.groupby(['user_id', 'day', 'hour', 'min_10']).size().reset_index().rename(columns={0:'min10_user_click'})
        min15_user_click = self.data.groupby(['user_id', 'day', 'hour', 'min_15']).size().reset_index().rename(columns={0:'min15_user_click'})
        min30_user_click = self.data.groupby(['user_id', 'day', 'hour', 'min_30']).size().reset_index().rename(columns={0:'min30_user_click'})
        min45_user_click = self.data.groupby(['user_id', 'day', 'hour', 'min_45']).size().reset_index().rename(columns={0:'min45_user_click'})

        self.data = pd.merge(self.data, min10_user_click, 'left', on=['user_id', 'day', 'hour', 'min_10'])
        self.data = pd.merge(self.data, min15_user_click, 'left', on=['user_id', 'day', 'hour', 'min_15'])
        self.data = pd.merge(self.data, min30_user_click, 'left', on=['user_id', 'day', 'hour', 'min_30'])
        self.data = pd.merge(self.data, min45_user_click, 'left', on=['user_id', 'day', 'hour', 'min_45'])
        
        del self.data['min_10']
        del self.data['min_15']
        del self.data['min_30']
        del self.data['min_45']
        
        return "continue"

    @findExtraCols
    def diffWithLastView(self,grainedList = [
        "shop_id"
    ]):
        for grain in grainedList:
            self.data['context_timestamp_str'] = self.data['context_timestamp'].astype(str)
            user_time_join = self.data.groupby(['user_id',grain])['context_timestamp_str'].agg(lambda x:';'.join(x)).reset_index()
            user_time_join.rename(columns={'context_timestamp_str':'user_'+grain+'_'+'time_join'},inplace = True)
            self.data = pd.merge(self.data,user_time_join,on=['user_id',grain],how='left')
            name = 'user_'+grain+'_'+'time_join'
            def f(x):
                if ( str(x.context_timestamp) == x[name] ):
                    return(-1) 
                else:
                    templist =  x[name].split(';')
                    for i in range(len(templist)-1,-1,-1):
                        if float(templist[i]) >= x.context_timestamp:
                            break
                    flag = i + 1
                    if flag > len(templist)-1 :
                        return  -1
                    else:
                        return x.context_timestamp - float( templist[flag] )


            self.data['diffWithLastView_'+grain] = self.data.apply(f,axis = 1)
            self.data.pop('context_timestamp_str')
            self.data.pop('user_'+grain+'_'+'time_join')
        return "continue"