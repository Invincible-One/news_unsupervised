import os
import datetime
import hashlib


def get_datetime_filename(filename_fmt, datetime_fmt="%m%d_%H%M"):
    current = datetime.datetime.now()
    current = current.strftime(datetime_fmt)
    return filename_fmt.format(current)



def hash_encoder(s):
    return hashlib.sha256(s.encode()).hexdigest()
