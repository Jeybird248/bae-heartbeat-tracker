# Heartbeat Tracker

An application used to train and test a model that can recognize a series of heartbeat readings, built with Python.Inspiration and credit for the original idea goes to [this post on Reddit by u/Loweren](https://www.reddit.com/r/Hololive/comments/owutig/oc_i_learned_programming_so_i_could_graph_inas/).

# Project Screen Shot(s)

![alt text](https://github.com/Jeybird248/bae-heartbeat-tracker/blob/main/images/training_image.png "Training Process")

![alt text](https://github.com/Jeybird248/bae-heartbeat-tracker/blob/main/output/results.png "Results")

# Installation and Setup Instructions

You will need the following libraries:

> os
> sys
> numpy
> cv2
 

# Reflection

This was a relatively short 1 week long personal project built to utilize what I had learned about the OpenCV library. Project goals included using image processing and basic machine learning to recognize and process the heartbeat reading images.

Originally I wanted to use an existing number classifying solution available on the Internet. What I found, however, was that the accuracy to which the numbers were being identified was around 50~60%, likely due to the close proximity of the characters with some combination of numbers. This is why I decided to train my own model using OpenCV's K-Nearest Neighbors model and creating a basic training application where the user could identify a series of numbers found using contours to train the model.

One of the main challenges I ran into was preparing the data itself. As stated above, the way that the readings were displayed made it difficult to recognize the number being shown without having number contours overlap or creating holes within the contours. This alongside the problem of the background sometimes being captured in the readings made for a lot of trial and error in finding the optimal setup that would give the correct reading every time.

In the next iteration I plan on using a more powerful model that can find the optimal image alteration settings and comprehend the overall shape of the number so that various fonts can be detected rather than just the one that the model was trained for. This can also allow for automatic crops for the reading images and even live graphing and quantification by accessing the live streams directly.
