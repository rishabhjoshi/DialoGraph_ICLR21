__author__ = 'anushabala'
import sys
sys.path.append('/usr1/home/rjoshi2/negotiation_personality/src/negotiation/bot/cocoa/src/basic/systems')
from system import System
from src.basic.sessions.human_session import HumanSession


class HumanSystem(System):
    def __init__(self):
        super(HumanSystem, self).__init__()

    @classmethod
    def name(cls):
        return 'human'

    def new_session(self, agent, kb):
        return HumanSession(agent)
