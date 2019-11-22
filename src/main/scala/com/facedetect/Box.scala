package com.facedetect

import java.lang.Math.max
import java.lang.Math.min
import java.awt.Point
import java.awt.Rectangle


class Box() {
  box = new Array[Int](4)
  bbr = new Array[Float](4)
  deleted = false
  landmark = new Array[Point](5)
  var box: Array[Int] = null //left:box[0],top:box[1],right:box[2],bottom:box[3]

  var score = .0 //probability

  var bbr: Array[Float] = null //bounding box regression

  var deleted = false
  var landmark: Array[Point] = null //facial landmark.只有ONet输出Landmark

  def left = box(0)

  def right = box(2)

  def top = box(1)

  def bottom = box(3)

  def width: Int = box(2) - box(0) + 1

  def height: Int = box(3) - box(1) + 1
/*
  //转为rect
  def transform2Rect: Rectangle = {
    val rect = new Rectangle()
    rect.height =
    rect.width  =
    rect.x      =
    rect.y      =
    rect.left = box(0)
    rect.top = box(1)
    rect.right = box(2)
    rect.bottom = box(3)
    rect
  }
*/
  //面积
  def area: Int = width * height
/*  //Bounding Box Regression
  def calibrate(): Unit = {
    val w = box(2) - box(0) + 1
    val h = box(3) - box(1) + 1
    box(0) = (box(0) + w * bbr(0)).toInt
    box(1) = (box(1) + h * bbr(1)).toInt
    box(2) = (box(2) + w * bbr(2)).toInt
    box(3) = (box(3) + h * bbr(3)).toInt
    var i = 0
    while ( {
      i < 4
    }) bbr(i) = 0.0f {
      i += 1; i - 1
    }
  }

  //当前box转为正方形
  def toSquareShape(): Unit = {
    val w = width
    val h = height
    if (w > h) {
      box(1) -= (w - h) / 2
      box(3) += (w - h + 1) / 2
    }
    else {
      box(0) -= (h - w) / 2
      box(2) += (h - w + 1) / 2
    }
  }

  //防止边界溢出，并维持square大小
  def limit_square(w: Int, h: Int): Unit = {
    if (box(0) < 0 || box(1) < 0) {
      val len = max(-box(0), -box(1))
      box(0) += len
      box(1) += len
    }
    if (box(2) >= w || box(3) >= h) {
      val len = max(box(2) - w + 1, box(3) - h + 1)
      box(2) -= len
      box(3) -= len
    }
  }

  def limit_square2(w: Int, h: Int): Unit = {
    if (width > w) box(2) -= width - w
    if (height > h) box(3) -= height - h
    if (box(0) < 0) {
      val sz = -box(0)
      box(0) += sz
      box(2) += sz
    }
    if (box(1) < 0) {
      val sz = -box(1)
      box(1) += sz
      box(3) += sz
    }
    if (box(2) >= w) {
      val sz = box(2) - w + 1
      box(2) -= sz
      box(0) -= sz
    }
    if (box(3) >= h) {
      val sz = box(3) - h + 1
      box(3) -= sz
      box(1) -= sz
    }
  }*/area
}