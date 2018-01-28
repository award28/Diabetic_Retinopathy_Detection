from django.shortcuts import render
from django.http import HttpResponse
from django.views import View


# Create your views here.
class Scan(View):
    template_name = 'scan/index.html'
    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)
    def post(self, request, *args, **kwargs):
        return HttpRequest("Nice post asshole. Now finish the view")
