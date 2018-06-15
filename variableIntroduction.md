
# Preprocessing:

## 1 _timePreprocessing:
    - 1 day
    - 2 hour
    - 3 minute
    - 4 maphour : cut hour into 4 bins
    - 5 mapmin  : cut minute into 4 bins( //15 min)

# FeatureEngineering

## _genExposeRatioThroughSth
    - ['user_id_hour_cnt', 'item_id_maphour_ratio', 'user_id_maphour_cnt', 'item_id_hour_cnt', 'item_id_maphour_cnt', 'item_id_hour_ratio', 'user_id_hour_ratio', 'user_id_maphour_ratio']

    - 统计每个用户、商品在不同的粒度的时间上浏览记录数

## _genAggregateLevelThroughSth
    - 统计用户、商品在某些维度上的平均水平
    - 比如第一个：统计每个商品上购买的用户的平均水平
        'item_id_TO_user_age_level_mean'
        'item_id_TO_hour_mean'
        'user_id_TO_hour_mean'
        'shop_id_TO_user_age_level_mean'
        'user_id_TO_user_age_level_mean'
        'item_brand_id_TO_user_age_level_mean'

## _genActivePerTimeGrain

    - 统计了比如item_brand_id 每天被多少个用户访问过
        'item_brand_id_day__active_user_id__num'
        'shop_id_day__active_user_id__num'
        'item_id_day__active_user_id__num'
        'item_category_list_day__active_item_id__num'
        'item_category_list_day__active_user_id__num'
        'user_id_day__active_item_city_id__num'
        'user_id_day__active_shop_id__num'
        'user_id_day__active_item_brand_id__num'

## _genRelativeRatioToAvg
    -   # 统计每个item_brand_id 的 price_level的均值
        # 并对与每一条观测的商品，查看其售价level和商铺的均值的比例（相对于商铺的其他商品的贵/便宜）
        'item_price_level_mean_by_item_brand_id'
        'item_sales_level_mean_by_item_brand_id'
        'item_collected_level_mean_by_item_brand_id'
        'item_price_level_mean_by_item_brand_idratio'
        'item_sales_level_mean_by_item_brand_idratio'
        'item_collected_level_mean_by_item_brand_idratio'
        'item_price_level_mean_by_item_category_list'
        'item_sales_level_mean_by_item_category_list'
        'item_collected_level_mean_by_item_category_list'
        'item_price_level_mean_by_item_category_listratio'
        'item_sales_level_mean_by_item_category_listratio'
        'item_collected_level_mean_by_item_category_listratio'
        'item_price_level_mean_by_item_city_id'
        'item_sales_level_mean_by_item_city_id'
        'item_collected_level_mean_by_item_city_id'
        'item_price_level_mean_by_item_city_idratio'
        'item_sales_level_mean_by_item_city_idratio'
        'item_collected_level_mean_by_item_city_idratio'


## doTricks1
    # 把用户的行为按照时间排序
    self.data.sort_values(['user_id','context_timestamp'],inplace =True)
    #统计每天用户第一次点击和最后一次点击的时间戳（by day）
        'click_user_id_tab'
        'click_item_id_user_id_tab'
        'click_item_brand_id_user_id_tab'
        'click_shop_id_user_id_tab'
        'click_item_city_id_user_id_tab'

## doTricks2
    # 统计点击时间差
    # 统计每个day下，item和第一个点击的时间差
        'i_day_diffTime_first'
        's_day_diffTime_first'
        'u_day_diffTime_first'
        'b_day_diffTime_first'
        'b_day_diffTime_last'
        's_day_diffTime_last'
        'i_day_diffTime_last'
        'u_day_diffTime_last'

## nexttimeDiff ,  lasttimeDiff
    # 统计具体时间差
    user_id_lasttime_diff                                   10000 non-null int64
    item_id_lasttime_diff                                   10000 non-null int64
    user_id_nexttime_diff                                   10000 non-null int64
    item_id_nexttime_diff                                   10000 non-null int64

## dorollWin
    #  查看用户购买的这些之间中，和这一次相近的（时间差小于600的次数）和与这一次不相近的次数
    #
    'user_count_10_bf'
    'user_count_10_af'
    'user_item_count_10_af'
    'user_shop_count_10_af'
    'user_item_count_10_bf'
    'user_shop_count_10_bf'

## doSize
    'min15_user_click'
    'min10_user_click'
    'min30_user_click'
    'min45_user_click'
    'user_id_query_day'
    'shop_item_unique_day'

## diffwithlastview
    'diffWithLastView_item_brand_id'
    'diffWithLastView_shop_id'
    'diffWithLastView_item_sales_level'