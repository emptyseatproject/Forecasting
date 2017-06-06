# save finalized model

#Todo
#Print date, predicted value, expected avlue. 17-03-19
#check how many rows should be deleted (sliding window)
#Check Correlation value 6.8
#Downsampling (from seconds to minutes) 7.1

# load finalized model and make a prediction
from pandas import Series
from statsmodels.tsa.arima_model import ARIMAResults
import numpy

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

series = Series.from_csv('dataset.csv')
months_in_year = 42
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(series.values, yhat, months_in_year)
print('Predicted: %.3f' % yhat)




# load and evaluate the finalized model on the validation dataset
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import TimeGrouper
import numpy
import time
import datetime
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load and prepare datasets
dataset = Series.from_csv('dataset.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 42
validation = Series.from_csv('validation-new.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
	# difference data
	months_in_year = 42
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(2,0,0))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = y[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)



start_time = datetime.datetime(2017, 4, 6, 6, 0)
###x = [start_time + datetime.timedelta(minutes = i*10) for i in range(len(y))]
###

#start_time = time.strptime("06:00:00", "%H:%M:%S")
#print(start_time)

#x1 = [start_time + datetime.timedelta(minutes = i*10) for i in range(42)]
print(len(y))
x1 = [start_time + datetime.timedelta(minutes = i*10) for i in range(len(y))]
x = [ ]
for i in range(len(y)):
	x.append(x1[i].time())
#x= x1
print(x)


#x = validation.index



pyplot.plot(x, predictions, color ='red', label ='Predicted number of available seats', linestyle='--', marker='x')
pyplot.plot(x, y, color='blue', label='Actual number of available seats', marker='8')

label_locations = [d for d in x if d.minute % 30 == 0 ]
labels = [d.strftime('%H:%M:%S') for d in label_locations]
#pyplot.xticks(x, rotation=90)
pyplot.xticks(label_locations, labels, rotation=90)
#pyplot.plot_date(x, y, fmt='M')
#pyplot.setp(pyplot.gca().xaxis.get_majorticklabels(),'rotation',90)

#pyplot.gcf().autofmt_xdate()
#pyplot.gca().set_xticks(x)

pyplot.legend()
pyplot.grid(True)
pyplot.title('Available seats on 5th bus stop')
pyplot.xlabel('Time')
pyplot.ylabel('Number of available seats')

pyplot.show()




'''
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy

# monkey patch around bug in ARIMA class
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))

ARIMA.__getnewargs__ = __getnewargs__

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load data
dataset = Series.from_csv('dataset.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 35
validation = Series.from_csv('validation.csv')
y = validation.values.astype('float32')

# difference data
months_in_year = 35
diff = difference(X, months_in_year)
# fit model
model = ARIMA(diff, order=(8, 0, 1))
model_fit = model.fit(trend='nc', disp=0)
# bias constant, could be calculated from in-sample mean residual
bias = -0.180524
# save model
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])

# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error

# rolling forecasts
for i in range(1, len(y)):
	# difference data
	months_in_year = 35
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(3,1,3))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = y[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)
pyplot.plot(y)
pyplot.plot(predictions, color='red')
pyplot.show()
'''
