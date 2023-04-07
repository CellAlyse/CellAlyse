# Advance usages

# Metriken
The area calculation is based on connected component labeling. This algorithm returns the `physical area`, which is not the actual area.
To obtaion the actual area I've introduced the `pixel pro µm` unit. 

*How to calculate `pixel pro µm`*
> You can either use a normal image. Normal means you know that the majority of the erythrocytes has a normal area and a **normal form**.
> Now you can calibrate `pixel pro µm` until the average area is about `50.2654824574`.
> If you don't have a reference image, you can open a python shell and get the `(x, y)` tuples of the outer points of a random cell.
> Now plug the values into this expression:
```python
int distance = (int) Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
```

*Calculating `pixel pro µm` using the website*
> Set pixel pro µm to 1.0 and upload an calibration image (std. diameter and **shape**). Divide the expected diameter with the diameter in pixel.
> If you don't have a normal cell, be aware that the main axis will be used.
