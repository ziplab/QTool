
1. FP16 training: Training with FP16 and SyncBN on multi-GPU seems to cause NAN loss for current projects (SyncBN option for FP16 is not finished). Use normal BN instead, currently.

2. Code might give some warnings, which would not cause any trouble for normal training and testing.

   ```
   Failing to import plugin, ModuleNotFoundError("No module named 'plugin'")
   loading third party model failed cannot import name 'model_zoo' from 'third_party' (unknown location)
   ```
   
3. Some words in the filename or config file are misspelled and we revise them anytime we found one. Thus, latency files might suffer from 'not-found' error. Check the filename if meeting such a situation. 

4. Pytorch version mismatch. Please upgrade pytorch version.

   a. error when loading pretrained model
   
   ```
   ValueError: invalid literal for int() with base 8: 'htq\x05ctor'
   ```
