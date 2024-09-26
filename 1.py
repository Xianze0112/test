PS C:\Users\gpu009> & c:/Users/gpu009/yunlian_doctor_venv/Scripts/python.exe c:/Users/gpu009/yunlian-doctr/My-DocTr-Plus-Yunlian/doctr_web.py     
Traceback (most recent call last):
  File "c:\Users\gpu009\yunlian-doctr\My-DocTr-Plus-Yunlian\doctr_web.py", line 62, in <module>
    model = model.cuda()
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\nn\modules\module.py", line 916, in cuda
    return self._apply(lambda t: t.cuda(device))
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\nn\modules\module.py", line 780, in _apply
    module._apply(fn)
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\nn\modules\module.py", line 780, in _apply
    module._apply(fn)
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\nn\modules\module.py", line 780, in _apply
    module._apply(fn)
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\nn\modules\module.py", line 805, in _apply
dules\module.py", line 805, in _apply
dules\module.py", line 805, in _apply
dules\module.py", line 805, in _apply
dules\module.py", line 805, in _apply
    param_applied = fn(param)
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\nn\modules\module.py", line 916, in <lambda>
dules\module.py", line 916, in <lambda>
    return self._apply(lambda t: t.cuda(device))
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\cuda\    return self._apply(lambda t: t.cuda(device))
    return self._apply(lambda t: t.cuda(device))
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\cuda\    return self._apply(lambda t: t.cuda(device))
    return self._apply(lambda t: t.cuda(device))
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\cuda\    return self._apply(lambda t: t.cuda(device))
  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\cuda\  File "C:\Users\gpu009\yunlian_doctor_venv\lib\site-packages\torch\cuda\__init__.py", line 305, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
PS C:\Users\gpu009>



AssertionError: Torch not compiled with CUDA enabled
PS C:\Users\gpu009>
AssertionError: Torch not compiled with CUDA enabled
AssertionError: Torch not compiled with CUDA enabled
PS C:\Users\gpu009>



                    & c:/Users/gpu009/yunlian_doctor_venv/Scripts/python.exe c:/Users/gpu009/yunlian-doctr/My-DocTr-Plus-Yunlian/doctr_web.py     
c:\Users\gpu009\yunlian-doctr\My-DocTr-Plus-Yunlian
True
absolute_path c:\Users\gpu009\yunlian-doctr\My-DocTr-Plus-Yunlian\model_save\model.pt
c:\Users\gpu009\yunlian-doctr\My-DocTr-Plus-Yunlian\doctr_web.py:40: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  pretrained_dict = torch.load(path, map_location=device)  # 将模型加载到
正确的设备上
 * Serving Flask app 'doctr_web'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://10.111.254.11:8080
Press CTRL+C to quit
 * Restarting with stat
c:\Users\gpu009\yunlian-doctr\My-DocTr-Plus-Yunlian
True
absolute_path c:\Users\gpu009\yunlian-doctr\My-DocTr-Plus-Yunlian\model_save\model.pt
c:\Users\gpu009\yunlian-doctr\My-DocTr-Plus-Yunlian\doctr_web.py:40: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrars possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  pretrained_dict = torch.load(path, map_location=device)  # 将模型加载到
正确的设备上
 * Debugger is active!
 * Debugger PIN: 380-378-948

