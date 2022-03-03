# PyData-Notes
Notes that I took on PyData Youtube Videos

### Video 1 - Sktime

[Video Link](https://www.youtube.com/watch?v=Wf2naBHRo8Q&ab_channel=PyData)

![Video 1 Cover](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/001/001_video_ss.png)


1) Some time series libraries + sktime are below.

![Time Series libraries](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/001/001_time_series.jpeg)

2) Tabularization can be visualized as follows:

![Tabularization](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/001/001_tabularization.jpeg)

### Video 2 - How to quickly build Data Pipelines for Data Scientists 


[Video Link](https://www.youtube.com/watch?v=XMnDCZhm9Go)

![Video 2 Cover](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/002/002_video_ss.png)

1) Some advices on data pipelining

![Time Series libraries](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/002/002_advices.png)

2) Parquet is a data format mostly used in cloud environment. It is easily convertable to csv's.


### Video 3 - Data Exploration Tools


[Video Link](https://www.youtube.com/watch?v=tiNQDY8ixXU)

![Video 3 Cover](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/003/003_video_ss.png)


1) Data Exploration Tools

* Open Source

    1) Pandas Profiling, Most famous 
    2) SweetVIZ, Target vs Input, creates HTML
    3) DABL, not developed anymore 
    4) dTreeViz , innovative to split data
    5) dtale, an interactive tool

* Commercial

    6) Trifacta
    7) SPSS

### Video 4 - Pandas Time Series Codes


[Video Link](https://www.youtube.com/watch?v=8upGdZMlkYM)

![Video 4 Cover](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/004/004_video_ss.png)

1) pd.date_range can be used to create datetime array(list with particular intervals)

```
# Daily date objects created and put in datetime_array. 366 elements in itself. Inclusive end.
import pandas as pd
datetime_array = pd.date_range(start ='1-1-2019',end ='1-1-2020', freq ='1D')
print(len(datetime_array))
```

2) Pandas `loc` attribute of dataframes or datetimes has a property of inclusive end. In python, end boundary are exclusive(for example: `range(0,10,1)`)

3) While plotting a line plot in matplotlib, reduce the size of dots via **linewidth** property

```
import pandas as pd
a = pd.Series([1,2,3])
a.plot(linewidth=0.5)
```

4) Seaborn boxplot is superior to matplotlib boxplot. It enables us to visualize grouped boxplots.

![Seaborn boxplot](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/004/004_seaborn.png)


5) A one dimensional time series data can be checked to have autocorrelation via autocorrelation_plot function.

```
import pandas as pd; import numpy as np
from pandas.profiling import autocorrelation_plot

a = pd.Series([1,2,3,4,1,1,1,1])

autocorrelation_plot(a)
```

![Autocorrelation plot](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/004/004_autocorrelation.png)

6) To fill absent values in a time series dataframe, under the condition that index is consisting of datetimes, use **asfreq** method([link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asfreq.html)).

7) Rolling mean of pandas react faster than weekly mean resample. If a bump and dump occurs in data consecutively, weekly mean resample neutralizes it but rolling mean was affected much more.

### Video 5 - Vincent D. Warmerdam: Untitled12.ipynb

![Video 5 Cover](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/005/005_video_ss.png)

1) Contextual help is a property of Jupyter lab, which splits the screen by two. It is showing us the definitions of methods when cursur is on top of the method.

![Contextual help](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/005/005_contextual_help.png)

2) Pandas group by can be used with `agg` method. In agg method, use kwargs(keyword arguments) that the key is the name of value we want to calculate. The value is a tuple consisting of 2 elements. First element is showing us on which column to apply aggregation and the second element is what aggregation to apply.

```
df = pd.DataFrame({
    'letter':['a','a','b','b','b','c','c'],
    'value': [1 ,5, 13, 12, 8, 3, 3 ]
})

df.groupby('letter').agg(
    unique_count = ('value','nunique'),
    mean_value = ('value',np.mean),
    max_value = ('value', max)
)

#         unique_count  mean_value  max_value
# letter                                     
# a                  2           3          5
# b                  3          11         13
# c                  1           3          3

```

3) pd.DataFrame().pipe(function_to_call) is allowing us to deal with pipelines similar to R.

```

df = pd.DataFrame({
    'A':[1,2,3,1,2,3,3,np.nan],
    'B':[10,10,10,20,30,50,np.nan,40]
})

def dropna_data(df):
    return df.dropna()

def calculate_max(df):
    return df.max(axis=1)

df.pipe(dropna_data).pipe(calculate_max)

```

4) Decorators are similar to packaging a gift in Python. They are functions that take functions as asgument and return functions. Used for checking(by 0 division etc, shape checking) and logging etc.

```

from datetime import datetime
import time

def mydecorator(func):
    def wrapper(func,*args, **kwargs):
        time_start= datetime.now()
        func()
        time_end = datetime.now()
        print(f"Time elapsed = {(time_end - time_start)}")
    return wrapper(func)

@mydecorator
def wait_3_seconds():
    time.sleep(3)

@mydecorator
def print_hello():
    print("hello world")

#Time elapsed = 0:00:03.003321
#hello world
#Time elapsed = 0:00:00.000074

```

5)  scikit-logo is a python framework aimimng to consolidate transformers(minmax, standard etc.), metrics and modules into a library that offer code quality/testing.

6) spaCy is an appealing python package in which there are sensible pipelines for NLP.

7) An example of Scikit Learn pipeline can be visualized as follows:

![Scikit pipeline](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/005/005_scikit_pipeline.jpeg)


8) Pipelines are enabling us to read them from left to right and top to bottom.

* Good pipeline:

    ![Good pipeline](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/005/005_good_pipeline.jpeg)

* Bad pipeline:

    ![Bad pipeline](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/005/005_bad_pipeline.jpeg)

* Crazy pipeline:

    ![Crazy pipeline](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/005/005_crazy_pipeline.jpeg)

9) [drawdata.xyz](https://drawdata.xyz) is website to visualize data that we created on GUI. It is useful for comprehending the problem.

