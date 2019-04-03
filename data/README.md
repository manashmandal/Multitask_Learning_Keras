# Data

You can download the images and labels from [Dataset](nju.edu.cn).
You'll then need to unrar the compessed dataset twice. The `processed.rar` file within the `miml-*.rar` file is the one you want.
Then you just need to rename the `*.mat` file to `dataset.h5` and put it in this `data/` directory.

Here's some pseudo code to try:

```bash
cd data
curl -O http://lamda.nju.edu.cn/data_MIMLimage.ashx?AspxAutoDetectCookieSupport=1)
unrar x *.rar
unrar x processed*.rar
mv *.mat dataset.h5
```
