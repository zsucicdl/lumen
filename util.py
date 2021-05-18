from datetime import datetime
import calendar

import numpy as np

def transform_order_date(order_date):
    #order_date = order_date_column.values
    order_date = order_date.apply(datetime.fromisoformat)
    day_of_week = []
    day_of_month = []
    day_of_year = []
    for date in order_date:
        dow = date.weekday()
        date_day = date.day-1
        doy = date.timetuple().tm_yday - 1 # starts from 1
        month_length = calendar.monthrange(date.year, date.month)[1]
        year_length = 365 if not calendar.isleap(date.year) else 366
        day_of_week.append(2*np.pi*dow / 7)
        day_of_month.append(2*np.pi*date_day / month_length)
        day_of_year.append(2*np.pi*doy / year_length)
    dow = np.array(day_of_week)
    dom = np.array(day_of_month)
    doy = np.array(day_of_year)
    return np.cos(dow), np.sin(dow), np.cos(dom), np.sin(dom), np.cos(doy), np.sin(doy)

def time_features(date):
#     print(type(date))
    dow_x, dow_y, dom_x, dom_y, doy_x, doy_y = transform_order_date(date['Order Date'])
    date['dow_x'] = dow_x
    date['dow_y'] = dow_y
    date['dom_x'] = dom_x
    date['dom_y'] = dom_y
    date['doy_x'] = doy_x
    date['doy_y'] = doy_y
    return date


log_norm_cols = ['Ordered qty', 'Invoiced price', 'Cost of part', '# of unique products on a quote']


class ClusterDropper:
    def fit(self, *_):
        return self
    
    def transform(self, x):
        y = x
        noto = (y['GM%'] > 1) | (y['GM%'] < -1)
        y = y[~noto]
        for feature_name in log_norm_cols:
            y = y.drop(index=y[(y[feature_name] <= 0)].index)
        return y


class Dropper():
    def fit(self, *_):
        return self

    def transform(self, x):
        return x.dropna()

