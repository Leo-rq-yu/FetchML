from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import matplotlib as plt
import plotly
import plotly.express as px
import json
import numpy as np

import fetchmloa as ml

app = Flask(__name__)

data = pd.read_csv(".\data_daily.csv")
data_testing = pd.DataFrame(np.log(data['Receipt_Count']).diff().diff(28))
global_ar_theta = np.ndarray((0,0))
global_ar_inter = np.ndarray((0,0))
global_ma_theta = np.ndarray((0,0))
global_ma_inter = np.ndarray((0,0))
global_data_c = pd.DataFrame()
global_res_c = pd.DataFrame()


# Route for "/" (frontend):
@app.route('/', methods=['POST','GET'])
def index():
	fig = px.line(data['Receipt_Count'])
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template("index.html", graphJSON = graphJSON)

@app.route('/adf_test', methods=['POST','GET'])
def abf_test_output():
        mes = ml.data_t(data)
        mes = mes.split('\n')
        return render_template("index.html", mes = mes)


@app.route('/acf_plot', methods=['POST','GET'])
def acf_plot():
	acf_fig, pacf_fig = ml.ACF_plot(data)
	acf_fig.savefig('static/acf_plot.png')
	pacf_fig.savefig('static/pacf_plot.png')
	graphs = True
	return render_template("index.html", graphs = graphs)

@app.route('/ml_plot', methods=['POST','GET'])
def ARMA():
	[data_train,data_test,theta,intercept,RMSE,ar_mes] = ml.AR(8,pd.DataFrame(data_testing['Receipt_Count']))
	data_c = pd.concat([data_train,data_test])
	global global_ar_theta
	global_ar_theta = theta
	global global_ar_inter
	global_ar_inter = intercept
	AR_fig = data_c[['Receipt_Count','Predicted_Values']].iloc[0:50].plot(figsize=(10,10))
	AR_fig.figure.savefig('static/AR_plot.png')
	res = pd.DataFrame()
	res['Residuals'] = data_c['Receipt_Count'] - data_c.Predicted_Values
	res_fig = res.plot(kind='kde')
	res_fig.figure.savefig('static/res_plot.png')
	[res_train,res_test,theta_res,intercept_res,RMSE,res_mes] = ml.MA(2,pd.DataFrame(res.Residuals))
	res_c = pd.concat([res_train,res_test])
	global global_ma_theta
	global_ma_theta = theta_res
	global global_ma_inter
	global_ma_inter = intercept_res
	data_c.Predicted_Values += res_c.Predicted_Values
	MA_fig = data_c[['Receipt_Count','Predicted_Values']].iloc[0:50].plot(figsize=(10,10))
	MA_fig.figure.savefig('static/MA_plot.png')
	trained = True
	data_c.Receipt_Count += pd.DataFrame(np.log(data['Receipt_Count'])).shift(2).Receipt_Count
	data_c.Receipt_Count += pd.DataFrame(np.log(data['Receipt_Count'])).diff().shift(8).Receipt_Count
	data_c.Predicted_Values += pd.DataFrame(np.log(data['Receipt_Count'])).shift(2).Receipt_Count 
	data_c.Predicted_Values += pd.DataFrame(np.log(data['Receipt_Count'])).diff().shift(8).Receipt_Count
	data_c.Receipt_Count = np.exp(data_c['Receipt_Count'])
	data_c.Predicted_Values = np.exp(data_c.Predicted_Values)
	global global_res_c
	global_res_c = res_c.copy()
	global global_data_c
	global_data_c = data_c.copy()
	fitted_fig = data_c[['Receipt_Count','Predicted_Values']].plot(figsize=(25,10))
	fitted_fig.figure.savefig('static/fitted.png')
	return render_template("index.html", Trained = trained, AR_mes = ar_mes, MA_res = res_mes)

@app.route('/fit_plot', methods=['POST','GET'])
def fitted():
        fit = True
        return render_template("index.html", Fitted = fit)
	

if __name__=='__main__':
        app.run(host='0.0.0.0', port=80,debug=True)