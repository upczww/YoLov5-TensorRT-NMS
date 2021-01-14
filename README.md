# yolov5
Original codes from [tensorrtx](https://github.com/wang-xinyu/tensorrtx). I modified the yololayer and integrated batchedNMSPlugin. A `yolov5s.wts` is provided for fast demo. How to generate `.wts` can refer to https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5.


## How to Run, yolov5s as example

1. build  and run
```
mkdir build
cd build
cmake ..
make
sudo ./yolov5 -s             // serialize model to plan file i.e. 'yolov5s.engine'
sudo ./yolov5 -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.
```
2. check the images generated, as follows. _zidane.jpg and _bus.jpg


<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247970-60b27c00-751e-11ea-88df-41473fed4823.jpg">
</p>

3. run Python example, please install Python tensorrt and Pycuda and then
```
python yolov5_trt.py
```
## More Information

See the readme in [tensorrtx home page.](https://github.com/wang-xinyu/tensorrtx)

## Known issues

None!