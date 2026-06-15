import SimpleITK as sitk
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import contextlib

def resample(image, transform,default_value=100):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    #interpolator = sitk.sitkCosineWindowedSinc
    interpolator = sitk.sitkLinear
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

def command_iteration(method):
    if (method.GetOptimizerIteration() == 0):
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")


def myreg(fixed,moving,extval):

  fixedim=sitk.GetImageFromArray(fixed)
  movingim=sitk.GetImageFromArray(moving)

  R = sitk.ImageRegistrationMethod()

  R.SetMetricAsCorrelation()

  R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,minStep=1e-4,numberOfIterations=500,gradientMagnitudeTolerance=1e-8)
  R.SetOptimizerScalesFromIndexShift()

  tx = sitk.CenteredTransformInitializer(fixedim, movingim,sitk.Similarity2DTransform())
  R.SetInitialTransform(tx)

  R.SetInterpolator(sitk.sitkLinear)

  R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
  with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    outTx = R.Execute(fixedim, movingim);

  resampledim = resample(movingim, outTx,extval)
  resampled=sitk.GetArrayViewFromImage(resampledim)
  
  return resampled



if __name__=='__main__':

  if len(sys.argv) < 3:
    print("Usage:", sys.argv[0],"<fixedImageFilter> <movingImageFile> ")
    sys.exit(1)

  pixelType = sitk.sitkFloat64

  fixed = sitk.ReadImage(sys.argv[1], pixelType)

  moving = sitk.ReadImage(sys.argv[2], pixelType)

  resampled=myreg(fixed,moving,255)

  plt.figure()
  array_view = sitk.GetArrayViewFromImage(moving)
  plt.imshow(array_view)

  plt.figure()
  array_view = sitk.GetArrayViewFromImage(fixed)
  plt.imshow(array_view,alpha=0.2)
  array_view = sitk.GetArrayViewFromImage(resampled)
  plt.imshow(array_view,alpha=0.2)
  plt.show()


#print(moving)

#sitk.WriteImage(moving, sys.argv[3])


#sitk.WriteTransform(outTx, sys.argv[3])
