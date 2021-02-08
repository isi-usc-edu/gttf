
import time


class WallClockTimer:
  
  def __init__(self):
    self.total_ = 0
    self.state_ = 0
    self.started_at_ = 0

  def start(self):
    if self.state_ != 0:
      raise ValueError('Each start() must be followed with stop() before next call to start()')
    self.state_ = 1
    self.started_at_ = time.time()

  def stop(self):
    now = time.time()
    if self.state_ != 1:
      raise valueerror('stop() must follow with start()')
    self.state_ = 0
    self.total_ += now - self.started_at_
  
  @property
  def total(self):
    if self.state_ != 0:
      raise valueerror('total must be called after stop()')
    
    return self.total_

