@echo on
 

set HPEESOF_DIR=C:\Program Files\Keysight\ADS2023
 

set path=%path%;%HPEESOF_DIR%\bin;%HPEESOF_DIR%\lib\win32_64;%HPEESOF_DIR%\circuit\lib.win32_64;%HPEESOF_DIR%\adsptolemy\lib.win32_64
set path=%path%;%HPEESOF_DIR%\fem\2023.00\win32_64\bin\tools\win32\python
set SIMARCH=win32_64
 
 

hpeesofsim --MTS_enabled  -r res.txt asmhemt.txt
