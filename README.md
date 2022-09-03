# Joint_Comma_and_Kasreh_Recognizer
This package includes source code for recognizing and adding Persian Kasre and Persian Comma to Persian text.
## Installation    
First, Clone the project,

```
git clone --single-branch --branch develop https://<token>@github.com/HRSadeghi/Joint_Comma_and_Kasreh_Recognizer.git
```
<br>
And then, install the project,

```
pip install .
```


## Usage

First, download the pretrained models,
```python
from Joint_kasreh_comma.download_models import *
downloader()
```

<br>
And then, use it like this,

```python
from Joint_kasreh_comma.joint_kasre_comma_recognizer import JointKasreCommaRecognizer

jkcr = JointKasreCommaRecognizer()
input_sen = 'خیر پس ازتردد در محدوده کنترل آلودگی هوا پلاک شما توسط دوربین ها ثبت می گردد و در ابتدا از سهمیه فصلی شما به صورت سیستمی کسر خواهد شد . در قسمت "خانه" می توانید سهمیه تردد باقیمانده در فصل را مشاهده نمایید و همچنین در قسمت "خدمات حمل و نقل" و "خودروی شخصی" تعداد روزهای تردد مجاز فصلی مصرف شده قابل رویت می باشد'

jkcr(input_sen)
```