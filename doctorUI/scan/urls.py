
from django.conf.urls import url, include

from . import views

urlpatterns = [
    url(r'^$', views.Scan.as_view(), name='scan')
]
