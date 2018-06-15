USER = "XUAN"
FILENAME = 'round1_ijcai_18_train_20180301.txt'
RUN_FIRST_TIME = False
COL_NEED_ENCODE = [
        
        'context_id', 'item_brand_id', 'item_city_id', 
        'item_id', 'user_id', 'shop_id',
        'context_page_id',  
        'user_occupation_id',
         
    ]

COLS_WITH_CATEGORY_TYPE =  ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id', 'context_page_id', 'shop_id', 'shop_review_num_level', 'shop_star_level', 'is_trade']
COLS_WITH_CONTINUE_TYPE =  ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
COLS_WITH_STR_TYPE = ['item_category_list', 'item_property_list', 'predict_category_property']
COLS_WITH_OTHER_TYPE =  ['context_timestamp']
COLS_THAT_USELESS = [
                        'context_timestamp','instance_id',
                        'item_id','user_id','context_id',
                        'context_page_id','shop_id','is_trade',
                        'item_category_list', 'item_property_list', 'predict_category_property'
                    ]
BEST_VARIABLE =  ['day','item_brand_id', 'item_sales_level', 'item_collected_level', 
'user_age_level', 'user_star_level', 'user_count_10_bf',
 'click_item_id_user_id_tab', 'u_day_diffTime_last', 'diffWithLastView_item_brand_id',
  'item_category_list_day__active_item_id__num', 'i_day_diffTime_last',
   'item_price_level_mean_by_item_brand_id', 'user_id_day__active_item_city_id__num',
    'user_id_lasttime_diff', 'i_day_diffTime_first', 
    'item_collected_level_mean_by_item_city_id']