# PyData-Notes
Notes that I took on PyData Youtube Videos

### Video 1 - Sktime

[Video Link](https://www.youtube.com/watch?v=Wf2naBHRo8Q&ab_channel=PyData)

![Video 1 Cover](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/001_video_ss.png)


1) Some time series libraries + sktime are below.

![Time Series libraries](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/001_time_series.jpeg)

2) Tabularization can be visualized as follows:

![Tabularization](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/001_tabularization.jpeg)

### Video 2 - How to quickly build Data Pipelines for Data Scientists 


[Video Link](https://www.youtube.com/watch?v=XMnDCZhm9Go)

![Video 2 Cover](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/002_video_ss.png)

1) Some advices on data pipelining

![Time Series libraries](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/002_advices.png)

2) Parquet is a data format mostly used in cloud environment. It is easily convertable to csv's.


### Video 3 - Data Exploration Tools


[Video Link](https://www.youtube.com/watch?v=tiNQDY8ixXU)

![Video 3 Cover](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/003_video_ss.png)


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

![Video 4 Cover](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/004_video_ss.png)

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

![Seaborn boxplot](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/004_seaborn.png)


5) A one dimensional time series data can be checked to have autocorrelation via autocorrelation_plot function.

```
import pandas as pd; import numpy as np
from pandas.profiling import autocorrelation_plot

a = pd.Series([1,2,3,4,1,1,1,1])

autocorrelation_plot(a)
```

![Autocorrelation plot](https://github.com/MuhammedBuyukkinaci/PyData-Notes/blob/master/images/004_autocorrelation.png)

6) To fill absent values in a time series dataframe, under the condition that index is consisting of datetimes, use **asfreq** method([link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asfreq.html)).

7) Rolling mean of pandas react faster than weekly mean resample. If a bump and dump occurs in data consecutively, weekly mean resample neutralizes it but rolling mean was affected much more.

