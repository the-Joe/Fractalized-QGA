#importing the libraries
#turtle standard graphics library for python
from turtle import *

#function to create koch koch_curve or koch curve
def koch_curve(lengthSide, levels):
    if levels == 0:
        forward(lengthSide)
        return
    lengthSide /= 3.0
    koch_curve(lengthSide, levels-1)
    left(60)
    koch_curve(lengthSide, levels-1)
    right(120)
    koch_curve(lengthSide, levels-1)
    left(60)
    koch_curve(lengthSide, levels-1)

#main function
if __name__ == "__main__":
    speed(0)                    #defining the speed of the turtle
    length = 300.0              
    penup()
    #Pull the pen up – no drawing when moving.
    #Move the turtle backward by distance, opposite to the direction the turtle is headed.
    #Do not change the turtle’s heading.
    backward(length/2.0)
    pendown()
    for i in range(3):
         #Pull the pen down – drawing when moving.
         koch_curve(length, 4)
         right(120)
    #To control the closing windows of the turtle
    mainloop()

"""
The Koch koch_curve (also known as the Koch curve, Koch star, or Koch island) is a mathematical curve and one of the earliest fractal curves
to have been described. It is based on the Koch curve, which appeared in a 1904 paper titled “On a continuous curve without tangents, 
constructible from elementary geometry” by the Swedish mathematician Helge von Koch.

How to construct one:

Step 1:
Draw an equilateral triangle. You can draw it with a compass or protractor, or just eyeball it if you don't want to spend too much time draw
ing the koch_curve. It's best if the length of the sides are divisible by 3, because of the nature of this fractal. This will become clear
in the next few steps.

Step 2:
Divide each side in three equal parts. This is why it is handy to have the sides divisible by three.

Step 3:
Draw an equilateral triangle on each middle part. Measure the length of the middle third to know the length of the sides of these new 
triangles.

Step 4:
Divide each outer side into thirds. You can see the 2nd generation of triangles covers a bit of the first. These three line segments
shouldn't be parted in three.

Step 5:
Draw an equilateral triangle on each middle part. Note how you draw each next generation of parts that are one 3rd of the mast one.

Step 6:
Repeat until you're satisfied with the amount of iterations. It will become harder and harder to accurately draw the new triangles, but 
with a fine pencil and lots of patience you can reach the 8th iteration. The one shown in the picture is a Koch koch_curve of the 4th 
iteration.

Step 7:
Decorate your koch_curve how you like it. You can colour it, cut it out, draw more triangles on the inside, or just leave it the way it is.
"""
