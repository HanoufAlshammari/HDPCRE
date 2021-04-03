from django.urls import path
from . import views

urlpatterns = [

    path('',views.home, name='home'),

    path('register/', views.registerPage, name="register"),
	path('login/', views.loginPage, name="login"),
	path('logout/', views.logoutUser, name="logout"),

    path('Home2',views.Home2, name='Home2'),
     path('result/', views.result, name='result'),
    ]
     #  path('',views.home, name='home'),
    #path('Home2',views.Home2, name='Home2'),
