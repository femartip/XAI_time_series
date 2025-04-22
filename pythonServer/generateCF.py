from pythonServer.NativeGuide.find_native_guide import find_native_cf
from pythonServer.NativeGuide.SubseqCF import find_subseq_cf

def generate_native_cf(ts, data_set_name, model_name):
    cf = find_native_cf(ts, data_set_name, model_name)
    return cf

def generate_subseq_cf(ts,data_set_name, model_name):
    cf = find_subseq_cf(ts,data_set_name, model_name)
    return cf