# 核心API  

算法核心是一个RootCore类，原则上它会占满分配的所有资源（所有CPU核心、指定的GPU）。  
因此资源调度需要外部虚拟化支持，比如K8S容器等，由虚拟化技术提供资源边界。  
算法核心提供的方法都是同步方法，应用可以用eventlet、celery等封装成异步调用。  
核心提供了辅助函数，用于在图片上画框和对应的分数。  
核心API输出错误的方式是抛出Error，应用代码可以try-excpet。  

## 一、初始化  

### 1.1 构造函数   

```py
root_core = RootCore(detection_model_path, score_model_path, private_key, device_name='cpu')
```

描述：  
初始化模型。  

参数：  
detection_model_path - 检测模型参数文件的路径。  
score_model_path - 评分模型参数文件的路径。  
private_key - 解密参数文件的密钥。模型参数要使用RSA-256公钥加密，私钥解密。  
device_name - 模型执行的资源，取值：'cpu', 'cuda:0', 'cuda:1'...，即所有CPU看作一个设备，每张显卡可以单独为一个设备，也可以用'cuda:0, 1'的形式指定多张显卡，或者直接用'cuda'指定所有显卡。    

返回值：   
root_core - 核心算法对象。  

## 二、高级API  

利用高级API可以一步完成整个评分流程。  

### 2.1 All-in-one评分  

```py
score_bboxes = root_core.score(picture_path)
```

描述：  
直接获取图片中所有已经治疗根管的位置和评分。  

参数：  
picture_path - 图片文件的路径，支持jpg，png。  

返回值：  
score_bboxes - List，其中每一项也是一个List，格式：  
```py
[x1, y1, x2, y2, score]
```

## 三、低级API  

利用低级API可以分步完成已治疗根管的检测、评分。  

### 3.1 根管检测  

```py
bboxes = root_core.detect(picture_path)
```

描述：  
获取图片中所有已经治疗根管的位置。  

参数：  
picture_path - 图片文件的路径，支持jpg，png。  

返回值：  
bboxes - List，包含图片中所有已经治疗根管的位置，其中每一项也是一个List，格式：  
```py
[x1, y1, x2, y2]
```

### 3.2 根管评分  

```py
score_bboxes = root_core.bbox_score(picture_path, bboxes)
```

描述：  
获取图片中bboxes对应的分数。  

参数：  
picture_path - 图片文件的路径，支持jpg，png。  
bboxes - 检测到的根管位置。是一个List，其中每一项也是一个List，格式：  
```py
[x1, y1, x2, y2]
```

返回值：
score_bboxes - List，其中每一项也是一个List，格式：  
```py
[x1, y1, x2, y2, score]
```

## 四、工具函数  

### 4.1 画框  

```py
draw_path = root_core.draw_score_bboxes(picture_path, result_prefix, score_bboxes, draw_score=True)
```

描述：  
将框和对应的分数叠加在图片上。  

参数：  
picture_path - 图片文件的路径，支持jpg，png。  
result_prefix - 输出文件名的构造规则：```img_path = '{}{}'.format(result_prefix, filename)```，filename只包括图片原文件名，不包含路径。  
score_bboxes - 检测到的根管位置及分数。是一个List，其中每一项也是一个List，格式：  
```py
[x1, y1, x2, y2, score]
```
result_path - 画框函数的输出路径。  
draw_score - 控制是否叠加分数的开关，如果设为False，则只画框，忽略score项（注意此时score_bboxes依然要符合前述的格式，score可以随意填入数字）。

返回值：  
draw_path - 已经叠加了框和分数的图片路径。  
