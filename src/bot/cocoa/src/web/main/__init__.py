__author__ = 'anushabala'
import sys
sys.path.append('/usr1/home/rjoshi2/negotiation_personality/src/negotiation/bot/cocoa/src/web/main')
from flask import Blueprint

main = Blueprint('main', __name__)
from . import routes, web_utils, backend
