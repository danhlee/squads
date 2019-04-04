from api import db
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


# create database model
class Match(db.Model):
  __tablename__ = "matches"
  id = db.Column(db.Integer, primary_key=True)
  b_top = db.Column(db.String(4))
  b_jung = db.Column(db.String(4))
  b_mid = db.Column(db.String(4))
  b_bot = db.Column(db.String(4))
  b_sup = db.Column(db.String(4))
  r_top = db.Column(db.String(4))
  r_jung = db.Column(db.String(4))
  r_mid = db.Column(db.String(4))
  r_bot = db.Column(db.String(4))
  r_sup = db.Column(db.String(4))
  winner = db.Column(db.String(3))
    
  def __init__(self,b_top,b_jung,b_mid,b_bot,b_sup,r_top,r_jung,r_mid,r_bot,r_sup,winner):
    self.b_top = b_top
    self.b_jung = b_jung
    self.b_mid = b_mid
    self.b_bot = b_bot
    self.b_sup = b_sup
    self.r_top = r_top
    self.r_jung = r_jung
    self.r_mid = r_mid
    self.r_bot = r_bot
    self.r_sup = r_sup
    self.winner = winner

  # __repr__ function for logging
  def __repr__(self):
    return "<Match '{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}'>".format(self.b_top,self.b_jung,self.b_mid,self.b_bot,self.b_sup,self.r_top,self.r_jung,self.r_mid,self.r_bot,self.r_sup,self.winner)
    # log = '<b_top %r>\n' % self.b_top
    # log += '<b_jung %r>\n' % self.b_jung
    # log += '<b_mid %r>\n' % self.b_mid
    # log += '<b_bot %r>\n' % self.b_bot
    # log += '<b_sup %r>\n' % self.b_sup
    # log += '<r_top %r>\n' % self.r_top
    # log += '<r_jung %r>\n' % self.r_jung
    # log += '<r_mid %r>\n' % self.r_mid
    # log += '<r_bot %r>\n' % self.r_bot
    # log += '<r_sup %r>\n' % self.r_sup
    # log += '<winner %r>\n' % self.winner
    # return log