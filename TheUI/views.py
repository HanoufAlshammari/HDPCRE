
from django.shortcuts import render,redirect
import pickle
# Create your views here.
from django.http import HttpResponse
# addit1
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
#from django.contrib.auth.decorators import login_required

from .forms import CreateUserForm
import joblib
import pandas as pd
import numpy as np


def home(request):
    return render(request,'home.html')
def Home2(request):
    return render(request,'Home2.html')





def getPredictions(agex, resting_blood_pressurex, sexx, cholesterolx, max_heart_rate_achievedx, st_depressionx, chest_pain_typex, fasting_blood_sugarx, rest_ecgx ,exercise_induced_anginax,
st_slopex, num_major_vesselsx, thalassemiax):
  col = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression',
              'sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg',
          'exercise_induced_angina', 'st_slope', 'num_major_vessels', 'thalassemia']



        #column names after one hot encoding
  one_hot_col = ['age', 'resting_blood_pressure', 'cholesterol',
          'max_heart_rate_achieved', 'st_depression', 'sex_female', 'sex_male','num_major_vessels_Zero',
          'num_major_vessels_One','num_major_vessels_Two','num_major_vessels_Three',
          'chest_pain_type_asymptomatic',
          'chest_pain_type_atypical angina', 'chest_pain_type_non-anginal pain',
          'chest_pain_type_typical angina',
          'fasting_blood_sugar_greater than 120mg/ml',
          'fasting_blood_sugar_lower than 120mg/ml',
          'rest_ecg_ST-T wave abnormality',
          'rest_ecg_left ventricular hypertrophy', 'rest_ecg_normal',
          'exercise_induced_angina_no', 'exercise_induced_angina_yes',
          'st_slope_downsloping', 'st_slope_flat', 'st_slope_upsloping',
          'thalassemia_fixed defect', 'thalassemia_normal',
          'thalassemia_reversable defect']

        #categorical = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg',
        #  'exercise_induced_angina', 'st_slope', 'thalassemia']
  input_df = pd.DataFrame(columns=one_hot_col)


        #numerical columns
  numerical = ['age', 'resting_blood_pressure', 'cholesterol',
        'max_heart_rate_achieved', 'st_depression']


  input_df = input_df.append({'age':agex,'resting_blood_pressure':resting_blood_pressurex,
  'cholesterol': cholesterolx,'max_heart_rate_achieved': max_heart_rate_achievedx, 'st_depression':st_depressionx},ignore_index=True)

        #load the model
  model = joblib.load('HD_classifier_model_rf.pkl')
  scaler = joblib.load('scaler.gz')


  #input_df[numerical] = scaler.fit(input_df[numerical]).transform(input_df[numerical])‏
  input_df[numerical] = scaler.transform(input_df[numerical])


  if (sexx == 'male'):
      input_df['sex_female'] = 0
      input_df['sex_male'] = 1

  elif (sexx == 'female'):
      input_df['sex_female'] = 1
      input_df['sex_male'] = 0


  if (chest_pain_typex == 'asymptomatic'):
      input_df['chest_pain_type_asymptomatic'] = 1
      input_df['chest_pain_type_atypical angina'] = 0
      input_df['chest_pain_type_non-anginal pain'] = 0
      input_df['chest_pain_type_typical angina'] = 0



  elif (chest_pain_typex == 'atypical angina'):
      input_df['chest_pain_type_asymptomatic'] = 0
      input_df['chest_pain_type_atypical angina'] = 1
      input_df['chest_pain_type_non-anginal pain'] = 0
      input_df['chest_pain_type_typical angina'] = 0



  elif (chest_pain_typex == 'non-anginal pain'):
      input_df['chest_pain_type_asymptomatic'] = 0
      input_df['chest_pain_type_atypical angina'] = 0
      input_df['chest_pain_type_non-anginal pain'] = 1
      input_df['chest_pain_type_typical angina'] = 0



  elif (chest_pain_typex == 'typical angina'):
      input_df['chest_pain_type_asymptomatic'] = 0
      input_df['chest_pain_type_atypical angina'] = 0
      input_df['chest_pain_type_non-anginal pain'] = 0
      input_df['chest_pain_type_typical angina'] = 1


  if (fasting_blood_sugarx == 'yes'):
      input_df['fasting_blood_sugar_lower than 120mg/ml'] = 0
      input_df['fasting_blood_sugar_greater than 120mg/ml'] = 1


  elif (fasting_blood_sugarx == 'no'):
      input_df['fasting_blood_sugar_lower than 120mg/ml'] = 1
      input_df['fasting_blood_sugar_greater than 120mg/ml'] = 0



  if (rest_ecgx =='Normal'):
      input_df['rest_ecg_ST-T wave abnormality']= 0
      input_df['rest_ecg_left ventricular hypertrophy']= 0
      input_df['rest_ecg_normal']= 1


  elif (rest_ecgx == 'st-t wave abnormality'):
      input_df['rest_ecg_ST-T wave abnormality']= 1
      input_df['rest_ecg_left ventricular hypertrophy']= 0
      input_df['rest_ecg_normal']= 0


  elif (rest_ecgx == 'rest_ecg_left ventricular hypertrophy'):
      input_df['rest_ecg_ST-T wave abnormality']= 0
      input_df['rest_ecg_left ventricular hypertrophy']= 1
      input_df['rest_ecg_normal']= 0



  if (exercise_induced_anginax == 'yes'):
      input_df[ 'exercise_induced_angina_no'] = 0
      input_df[ 'exercise_induced_angina_yes'] = 1


  elif (exercise_induced_anginax == 'no'):
      input_df['exercise_induced_angina_no'] = 1
      input_df['exercise_induced_angina_yes'] = 0



  if (st_slopex == 'flat'):
      input_df['st_slope_upsloping']= 0
      input_df['st_slope_downsloping']= 0
      input_df['st_slope_flat']= 1


  elif (st_slopex  == 'st_slope_upsloping'):
      input_df['st_slope_upsloping']= 1
      input_df['st_slope_downsloping']= 0
      input_df['st_slope_flat']= 0


  elif (st_slopex == 'st_slope_downsloping'):
      input_df['st_slope_upsloping']= 0
      input_df['st_slope_downsloping']= 1
      input_df['st_slope_flat']= 0


  if (num_major_vesselsx == 'Zero'):
      input_df['num_major_vessels_Zero'] = 1
      input_df['num_major_vessels_One'] = 0
      input_df['num_major_vessels_Two'] = 0
      input_df['num_major_vessels_Three'] = 0


  elif (num_major_vesselsx == 'One'):
      input_df['num_major_vessels_Zero'] = 0
      input_df['num_major_vessels_One'] = 1
      input_df['num_major_vessels_Two'] = 0
      input_df['num_major_vessels_Three'] = 0


  elif (num_major_vesselsx == 'Two'):
      input_df['num_major_vessels_Zero'] = 0
      input_df['num_major_vessels_One'] = 0
      input_df['num_major_vessels_Two'] = 1
      input_df['num_major_vessels_Three'] = 0


  elif (num_major_vesselsx == 'Three'):
      input_df['num_major_vessels_Zero'] = 0
      input_df['num_major_vessels_One'] = 0
      input_df['num_major_vessels_Two'] = 0
      input_df['num_major_vessels_Three'] = 1



  if (thalassemiax =='normal'):
      input_df['thalassemia_fixed defect']= 0
      input_df['thalassemia_reversable defect'] = 0
      input_df['thalassemia_normal']= 1


  elif (thalassemiax  == 'thalassemia_fixed defect'):
      input_df['thalassemia_fixed defect']= 1
      input_df['thalassemia_reversable defect']= 0
      input_df['thalassemia_normal']= 0


  elif( thalassemiax == 'thalassemia_reversable defect'):
      input_df['thalassemia_fixed defect']= 0
      input_df['thalassemia_reversable defect'] = 1
      input_df['thalassemia_normal']= 0



  prediction = model.predict(input_df)

  if prediction == 0:
      return 'The patient is more likely not to have a heart disease'
  elif prediction == 1:
      return 'The patient is more likely to have a heart disease'
  else:
      return 'error'




def result(request):
    #age = None
    #resting_blood_pressure = None
    #sex = None
    #cholesterol = None
    #max_heart_rate_achieved = None
    #st_depression = None
    #chest_pain_type = None
    #fasting_blood_sugar = None
    #rest_ecg = None
    #exercise_induced_angina = None
    #st_slope = None
    #num_major_vessels = None
    #thalassemia = None

    # take input from user
    if request.method == 'POST':
        age=request.POST.get('age')
        #age= (request.POST.get['age'])

        #age =(request.POST.get['age'])
        resting_blood_pressure = request.POST.get('resting_blood_pressure')
        sex = request.POST.get('sex')
        cholesterol = request.POST.get('cholesterol')
        max_heart_rate_achieved = request.POST.get('max_heart_rate_achieved')
        st_depression = request.POST.get('st_depression')
        chest_pain_type = request.POST.get('chest_pain_type')
        fasting_blood_sugar = request.POST.get('fasting_blood_sugar')
        rest_ecg = request.POST.get('rest_ecg')
        exercise_induced_angina = request.POST.get('exercise_induced_angina')
        st_slope = request.POST.get('st_slope')
        num_major_vessels = request.POST.get('num_major_vessels')
        thalassemia = request.POST.get('thalassemia')
        print(age, resting_blood_pressure,sex, cholesterol, max_heart_rate_achieved, st_depression, chest_pain_type, fasting_blood_sugar,rest_ecg, exercise_induced_angina, st_slope, num_major_vessels,thalassemia)






            #global age = (request.GET['age'])




        #resting_blood_pressure = int(request.GET['resting_blood_pressure'])
        #sex = str(request.GET['sex'])
        #cholesterol = int(request.GET['cholesterol'])
        #max_heart_rate_achieved = int(request.GET['max_heart_rate_achieved'])
        #st_depression = int(request.GET['st_depression'])
        #chest_pain_type = str(request.GET['chest_pain_type'])
        #fasting_blood_sugar = str(request.GET['fasting_blood_sugar'])
        #rest_ecg = str(request.GET['rest_ecg'])
        #exercise_induced_angina = str(request.GET['exercise_induced_angina'])
        #st_slope = str(request.GET['st_slope'])
        #num_major_vessels = int(request.GET['num_major_vessels'])
        #thalassemia = str(request.GET['thalassemia'])


    result = getPredictions(age,resting_blood_pressure,sex,cholesterol,max_heart_rate_achieved,st_depression,chest_pain_type,fasting_blood_sugar,rest_ecg,exercise_induced_angina,st_slope,num_major_vessels,thalassemia)

    return render(request, 'result.html', {'result': result})


#input proccesing
    #columns_to_scale = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
#def scale_numerical(data,columns_to_scale):
 # from sklearn.preprocessing import StandardScaler
  #standardScaler = StandardScaler()
  #data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])
  #return data ‏

#get predection





# addit2

def registerPage(request):
	#if request.user.is_authenticated:
		#return redirect('home')
	#else:
		form = CreateUserForm()
		if request.method == 'POST':
			form = CreateUserForm(request.POST)
			if form.is_valid():
				form.save()
				user = form.cleaned_data.get('username')
				messages.success(request, 'Account was created for ' + user)

				return redirect('login')


		context = {'form':form}
		return render(request, 'accounts/register.html', context)

def loginPage(request):
	#if request.user.is_authenticated:
		#return redirect('home')
	#else:
		if request.method == 'POST':
			username = request.POST.get('username')
			password =request.POST.get('password')

			user = authenticate(request, username=username, password=password)

			if user is not None:
				login(request, user)
				return redirect('Home2')
			else:
				messages.info(request, 'Username OR password is incorrect')

		context = {}
		return render(request, 'accounts/login.html', context)

def logoutUser(request):
	logout(request)
	return redirect('login')
