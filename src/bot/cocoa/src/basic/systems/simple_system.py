__author__ = 'anushabala'
from system import System
from src.basic.sessions.simple_session import SimpleSession
from src.basic.sessions.timed_session import TimedSessionWrapper


class SimpleSystem(System):
    def __init__(self, lexicon, timed_session=False, consecutive_entity=True, realizer=None, strat_model = 'graph'):
        super(SimpleSystem, self).__init__()
        self.lexicon = lexicon
        self.timed_session = timed_session
        self.consecutive_entity = consecutive_entity
        self.realizer = realizer
        self.strat_model = strat_model

    @classmethod
    def name(cls):
        return 'simple'

    def new_session(self, agent, kb, style):
        session = SimpleSession(agent, kb, self.lexicon, style, self.realizer, self.consecutive_entity, strat_model = self.strat_model)
        if self.timed_session:
            session = TimedSessionWrapper(agent, session)
        return session
