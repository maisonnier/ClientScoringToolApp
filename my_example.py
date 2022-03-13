from flask import Flask,render_template,session,url_for,redirect, request
import numpy as np
import pandas as pd
from flask_wtf import FlaskForm
# from wtforms import TextField, SubmitField
from wtforms import StringField, SubmitField
from lightgbm import LGBMClassifier
# import joblib
import pickle


import base64
from io import BytesIO
from matplotlib.figure import Figure
import numpy as np

with open('model_lgbm_12var.pkl', 'rb') as inp:
	my_model_load = pickle.load(inp)

model = my_model_load[0]
model_params = my_model_load[1]
X_all_no_nan = my_model_load[2]
y_all_no_nan = my_model_load[3]
df_recall = my_model_load[4]

list_cut_off_input = list(df_recall.cut_off.values)

list_recall_input = list(df_recall.recall.values)

params_client = np.zeros(14)

client_proba = 0


def return_prediction(model,sample_json):
	client_recall = sample_json['client_recall']
	EXT_SOURCE_3 = sample_json['EXT_SOURCE_3']
	EXT_SOURCE_1 = sample_json['EXT_SOURCE_1']
	EXT_SOURCE_2 = sample_json['EXT_SOURCE_2']
	AMT_ANNUITY = sample_json['AMT_ANNUITY']

	DAYS_BIRTH = sample_json['DAYS_BIRTH']
	AMT_CREDIT = sample_json['AMT_CREDIT']
	DAYS_LAST_PHONE_CHANGE = sample_json['DAYS_LAST_PHONE_CHANGE']
	AMT_GOODS_PRICE = sample_json['AMT_GOODS_PRICE']

	DAYS_ID_PUBLISH = sample_json['DAYS_ID_PUBLISH']
	DAYS_EMPLOYED = sample_json['DAYS_EMPLOYED']
	AMT_INCOME_TOTAL = sample_json['AMT_INCOME_TOTAL']
	OWN_CAR_AGE = sample_json['OWN_CAR_AGE']

	credit_info = [[EXT_SOURCE_3, EXT_SOURCE_1, EXT_SOURCE_2, AMT_ANNUITY,\
									DAYS_BIRTH, AMT_CREDIT, DAYS_LAST_PHONE_CHANGE, AMT_GOODS_PRICE,\
									DAYS_ID_PUBLISH, DAYS_EMPLOYED, AMT_INCOME_TOTAL, OWN_CAR_AGE]]

	params_client[12] = sample_json['client_recall']
	params_client[0] = sample_json['EXT_SOURCE_3']
	params_client[1] = sample_json['EXT_SOURCE_1']
	params_client[2] = sample_json['EXT_SOURCE_2']
	params_client[3] = sample_json['AMT_ANNUITY']
	params_client[4] = sample_json['DAYS_BIRTH']
	params_client[5] = sample_json['AMT_CREDIT']
	params_client[6] = sample_json['DAYS_LAST_PHONE_CHANGE']
	params_client[7] = sample_json['AMT_GOODS_PRICE']
	params_client[8] = sample_json['DAYS_ID_PUBLISH']
	params_client[9] = sample_json['DAYS_EMPLOYED']
	params_client[10] = sample_json['AMT_INCOME_TOTAL']
	params_client[11] = sample_json['OWN_CAR_AGE']


	classes = np.array(['approved', 'rejected'])

	y_pred_proba = model.predict_proba(credit_info)[:, 1][0]

	client_cutoff = list_cut_off_input[min(range(len(list_recall_input)), key=lambda i: abs(list_recall_input[i] - params_client[12]))]

	if y_pred_proba > client_cutoff:
		y_pred = 1
	else:
		y_pred = 0

	return classes[y_pred], y_pred_proba


def ft_graph_histogram(X_train, selected_param, val_client):
	fig = Figure()
	ax = fig.subplots()
	ax.hist(X_train[selected_param])
	ax.axvline(x= val_client, color='magenta', label='Customer')
	# myTitle = "Chosen parameter "+str(selected_param)+" Client (purple) vs other clients histogram"
	# fig.title(" Client (purple) vs other clients histogram")
	# ax.title()(" Client (purple) vs other clients histogram")
	# ax.xlabel(selected_param+ " values")
	# ax.ylabel("Number of clients")
	# Save it to a temporary buffer.
	buf = BytesIO()
	fig.savefig(buf, format="png")
	# Embed the result in the html output.
	data = base64.b64encode(buf.getbuffer()).decode("ascii")
	return data

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class CreditForm(FlaskForm):
	client_recall = StringField('client_recall', default = "0.90")
	EXT_SOURCE_3 = StringField('EXT_SOURCE_3', default = "0.130947")
	EXT_SOURCE_1 = StringField('EXT_SOURCE_1', default = "0.048507")
	EXT_SOURCE_2 = StringField('EXT_SOURCE_2', default = "0.472275")
	AMT_ANNUITY = StringField('AMT_ANNUITY', default = "6750.0")
	DAYS_BIRTH = StringField('DAYS_BIRTH', default = "-14186")
	AMT_CREDIT = StringField('AMT_CREDIT', default = "135000.0")
	DAYS_LAST_PHONE_CHANGE = StringField('DAYS_LAST_PHONE_CHANGE', default = "-576.0")
	AMT_GOODS_PRICE = StringField('AMT_GOODS_PRICE', default = "135000.0")
	DAYS_ID_PUBLISH = StringField('DAYS_ID_PUBLISH', default = "-5024.0")
	DAYS_EMPLOYED = StringField('DAYS_EMPLOYED', default = "-1614")
	AMT_INCOME_TOTAL = StringField('AMT_INCOME_TOTAL', default = "63000.0")
	OWN_CAR_AGE = StringField('OWN_CAR_AGE', default = "14.0")

	submit = SubmitField("Analyze Client Data")





@app.route("/",methods=['GET','POST'])
@app.route("/home",methods=['GET','POST'])
def index():

	form = CreditForm()

	if form.validate_on_submit():

		session['client_recall'] = form.client_recall.data
		session['EXT_SOURCE_3'] = form.EXT_SOURCE_3.data
		session['EXT_SOURCE_1'] = form.EXT_SOURCE_1.data
		session['EXT_SOURCE_2'] = form.EXT_SOURCE_2.data
		session['AMT_ANNUITY'] = form.AMT_ANNUITY.data
		session['DAYS_BIRTH'] = form.DAYS_BIRTH.data
		session['AMT_CREDIT'] = form.AMT_CREDIT.data
		session['DAYS_LAST_PHONE_CHANGE'] = form.DAYS_LAST_PHONE_CHANGE.data
		session['AMT_GOODS_PRICE'] = form.AMT_GOODS_PRICE.data
		session['DAYS_ID_PUBLISH'] = form.DAYS_ID_PUBLISH.data
		session['DAYS_EMPLOYED'] = form.DAYS_EMPLOYED.data
		session['AMT_INCOME_TOTAL'] = form.AMT_INCOME_TOTAL.data
		session['OWN_CAR_AGE'] = form.OWN_CAR_AGE.data

		return redirect(url_for("prediction"))

	return render_template('home.html',form=form)






@app.route('/prediction')
def prediction():

	content = {}

	content['client_recall'] = float(session['client_recall'])
	content['EXT_SOURCE_3'] = float(session['EXT_SOURCE_3'])
	content['EXT_SOURCE_1'] = float(session['EXT_SOURCE_1'])
	content['EXT_SOURCE_2'] = float(session['EXT_SOURCE_2'])
	content['AMT_ANNUITY'] = float(session['AMT_ANNUITY'])
	content['DAYS_BIRTH'] = float(session['DAYS_BIRTH'])
	content['AMT_CREDIT'] = float(session['AMT_CREDIT'])
	content['DAYS_LAST_PHONE_CHANGE'] = float(session['DAYS_LAST_PHONE_CHANGE'])
	content['AMT_GOODS_PRICE'] = float(session['AMT_GOODS_PRICE'])
	content['DAYS_ID_PUBLISH'] = float(session['DAYS_ID_PUBLISH'])
	content['DAYS_EMPLOYED'] = float(session['DAYS_EMPLOYED'])
	content['AMT_INCOME_TOTAL'] = float(session['AMT_INCOME_TOTAL'])
	content['OWN_CAR_AGE'] = float(session['OWN_CAR_AGE'])

	client_results, client_proba = return_prediction(model,content)

	# client_cut_off = 0.29
	client_recall = content['client_recall']

	client_cutoff = list_cut_off_input[min(range(len(list_recall_input)), key=lambda i: abs(list_recall_input[i] - content['client_recall']))]

	# Generate the figure **without using pyplot**.
	fig = Figure()
	ax = fig.subplots()
	ax.scatter(list_cut_off_input, list_recall_input)
	ax.scatter(client_cutoff, client_recall, color = 'magenta')
	# Save it to a temporary buffer.
	buf = BytesIO()
	fig.savefig(buf, format="png")
	# Embed the result in the html output.
	data = base64.b64encode(buf.getbuffer()).decode("ascii")
	return render_template('prediction.html',results=client_results, client_proba=round(client_proba,2), data=data, client_cutoff=client_cutoff)


@app.route('/analyse',methods = ['GET'])
def analyse():

	return render_template('analyse.html')

@app.route("/submitted", methods = ['POST'])
def selection_param():
	selected_param = request.form.get('select_client_param')
	print(selected_param)
	# return select

	index_param = model_params.index(selected_param)
	val_client = params_client[index_param]
	data = ft_graph_histogram(X_all_no_nan, selected_param, val_client)
	# return f"<img src='data:image/png;base64,{data}'/>"
	return render_template('myplot.html',data=data, selected_param=selected_param)




if __name__=='__main__':
	app.run()
