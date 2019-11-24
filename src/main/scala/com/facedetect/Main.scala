package com.facedetect

//import libraries
import java.nio.FloatBuffer
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.awt.Image
import java.util._  // This imports all
import java.nio.file.{Files, Path, Paths}
import java.lang.Math.{max,min}
import javax.imageio.ImageIO
import util.control.Breaks._
import java.util.Base64 //Might not be needed since i have imported all utils.
//import java.util.Arrays.toString //Might not be needed.
//import java.awt.Graphics2D

import org.tensorflow.op.core.Log
import org.tensorflow.{Graph, Session, Tensor}


case class TensorflowObject(graph: Graph, sess: Session)

class Mtcnn{
  // parameter
  private val factor = 0.709f
  private val PNetThreshold = 0.6f
  private val RNetThreshold = 0.7f
  private val ONetThreshold = 0.7f


  private val graph = new Graph

  //MODEL PATH
  private val MODEL_FILE = "mtcnn_freezed_model.pb"

  //tensor name
  private val PNetInName = "pnet/input:0"
  private val PNetOutName = Array[String]("pnet/prob1:0", "pnet/conv4-2/BiasAdd:0")
  private val RNetInName = "rnet/input:0"
  private val RNetOutName = Array[String]("rnet/prob1:0", "rnet/conv5-2/conv5-2:0")
  private val ONetInName = "onet/input:0"
  private val ONetOutName = Array[String]("onet/prob1:0", "onet/conv6-2/conv6-2:0", "onet/conv6-3/conv6-3:0")

  //Loading graph
  def loadGraph(modelBytes: Array[Byte]): TensorflowObject = {
    val graph = new Graph()
    graph.importGraphDef(modelBytes)
    val sess = new Session(graph)
    TensorflowObject(graph, sess)
  }
  /*
    //Another that works also
    def convertImageToBase64(imagePath: String): String = {
      val in = ImageIO.read(new File(imagePath))
      val outputStream = new ByteArrayOutputStream()
      ImageIO.write(in, "png", outputStream)
      val returnedString = Base64.getEncoder.encodeToString(outputStream.toByteArray)

    }*/
  //Converting image to base64
  def convertImageToBase64(imagePath: Path): String = {
    val f = Files.readAllBytes(imagePath)
    val encodedString = Base64.getEncoder.encodeToString(f)
    encodedString
  }


  //Converting base64 image into required buffered image
  def convertBase64ToBufferedImage(imageString: String): BufferedImage = {
    val image = ImageIO.read(new ByteArrayInputStream(Base64.getDecoder.decode(imageString)))
    image
    /* another method of conversion
      val decoder: BASE64Decoder = new BASE64Decoder
      val decodedBytes: Array[Byte] = decoder.decodeBuffer(imageString: String)

      val image: BufferedImage = ImageIO.read(new ByteArrayInputStream(decodedBytes))
      image
     */
  }
  //Converting image into required tensor format
  def convertBufferedImageToFloatBuffer(imageBuffer: BufferedImage): FloatBuffer = {
    val image = imageBuffer.asInstanceOf[BufferedImage]
    //val imagePixels = new Float()(photos.getWidth() * photos.getHeight() * 3)
    val h = image.getWidth()
    val w = image.getHeight()
    val channels = 3

    //Convert DocumentMatchImage to Tensor
    var index = 0
    val fb = FloatBuffer.allocate(w * h * channels)

    for (row <- 0 until h) {
      for (column <- 0 until w) {
        val pixel = image.getRGB(column, row)
        //Note that we used var here and we had to divide by 255 I need to know why
        val red: Float = (pixel >> 16) & 0xff
        val green: Float = (pixel >> 8) & 0xff
        System.exit(1)
        val blue: Float = pixel & 0xff
        //Scale DocumentMatchImage
        //        red = red / 255.0f
        //        green = green / 255.0f
        //        blue = blue / 255.0f
        fb.put(index, red)
        index += 1
        fb.put(index, green)
        index += 1
        fb.put(index, blue)
        index += 1
      }
    }
    println(java.util.Arrays.toString(fb.array()))
    fb
  }

  //Read Bitmap pixel values, preprocess (-127.5 /128), convert to one-dimensional array return
  def normalizeImage(imageBuffer: BufferedImage): FloatBuffer ={
    val image = imageBuffer.asInstanceOf[BufferedImage]
    //val imagePixels = new Float()(photos.getWidth() * photos.getHeight() * 3)
    val h = image.getHeight()
    val w = image.getWidth()
    val channels = 3
    //Convert DocumentMatchImage to Tensor
    var index = 0
    val fb = FloatBuffer.allocate(w * h * channels)
    val imageMean = 127.5f
    val imageStd = 128
    for (row <- 0 until h) {
      for (column <- 0 until w) {
        val pixel = image.getRGB(column, row)
        //Note that we used var here and we had to divide by 255 I need to know why
        val red: Float = (((pixel >> 16) & 0xff) - imageMean) / imageStd
        val green: Float = (((pixel >> 8) & 0xff) - imageMean) / imageStd
        val blue: Float = ((pixel & 0xff) - imageMean) / imageStd
        //Scale DocumentMatchImage
        //        red = red / 255.0f
        //        green = green / 255.0f
        //        blue = blue / 255.0f
        fb.put(index, red)
        index += 1
        fb.put(index, green)
        index += 1
        fb.put(index, blue)
        index += 1
      }
    }
    fb
  }



  //Detect face, minSize is the smallest face pixel value
  private def imageResize(imageBuffer: BufferedImage, scale: Float): BufferedImage = {
    val image = imageBuffer.asInstanceOf[BufferedImage]
    val height = image.getHeight()
    val width = image.getWidth()
    val newHeight: Int = (height * scale).toInt
    val newWidth: Int = (width * scale).toInt
    val tmp = image.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH)
    val dImg = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_ARGB)
    val g2d = dImg.createGraphics
    g2d.drawImage(tmp, 0, 0, null)
    g2d.dispose()
    dImg
  }
  //Flip alone diagonal
  //The diagonal is flipped. The data size was originally h*w*stride, and turned into w*h*stride after flipping.
  def flipDiag(data: Array[Float], h: Int, w: Int, stride: Int): Unit = {
    val tmp = new Array[Float](w * h * stride)
    var i = 0
    for( i <- 0 until (w*h*stride)){tmp(i) = data(i)}
    var y = 0
    for (y <- 0 until h) {
      var x = 0
      for (x <- 0 until w){
        var z = 0
        for (z <- 0 until stride){data((x * h + y) * stride + z) = tmp((y * w + x) * stride + z)}
      }
    }
  }

  //Src is converted to 2D and stored in dst
  def expand(src: Array[Float], dst: Array[Array[Float]]): Unit ={
    var idx = 0
    for (y <- 0 until  dst.length ){
      for (x <- 0 until  dst(0).length) {
        dst(y)(x) = src({idx += 1})
      }
    }
  }

  //Src is converted to 3D and stored in dst
  def expand(src: Array[Float], dst: Array[Array[Array[Float]]]): Unit = {
    var idx = 0
    for (y <- 0 until  dst.length){
      for (x <- 0 until  dst(0).length) {
        for (c <- 0 until dst(0)(0).length) {
          dst(y)(x)(c) = src({idx += 1})
        }
      }
    }
  }

  //dst=src[:,:,1]
  def expandProb(src: Array[Float], dst: Array[Array[Float]]): Unit = {
    var idx = 0
    for (y <- 0 until  dst.length){
      for (x <- 0 until  dst(0).length) {
        dst(y)(x) = src({idx += 1; idx - 1} * 2 + 1) // I am not sure about this , check and recheck abeg
      }
    }
  }

  //To flip before input, the output should also be flipped
  def pnetForward(imageBuffer: BufferedImage, session: Session, PNetOutProb: Array[Array[Float]], PNetOutBias: Array[Array[Array[Float]]]): Int = {
    val image = imageBuffer.asInstanceOf[BufferedImage]
    val height = image.getHeight()
    val width = image.getWidth()
    val PNetIn = normalizeImage(imageBuffer)
    val PNetInNew = normalizeImage(imageBuffer).array()
   // println(PNetIn)
    flipDiag(PNetInNew, height, width, 3) //Flip along the diagonal
    val PNetOutSizeW = Math.ceil(width * 0.5 - 5).toInt
    val PNetOutSizeH = Math.ceil(height * 0.5 - 5).toInt
    val PNetOutP = new Array[Float](PNetOutSizeW * PNetOutSizeH * 2)
    val PNetOutB = new Array[Float](PNetOutSizeW * PNetOutSizeH * 4)
    val shape = Array[Long](1, width, height, 3)
    val modelInput = Tensor.create(shape, PNetIn)
    val result = session.runner.feed(PNetInName, modelInput).fetch(PNetOutName(0)).run().get(0) //Index to get
    result.copyTo(PNetOutP)(0) //result.copyTo(Array.ofDim[Float](1, 512))
    val resultTwo = session.runner.feed(PNetInName, modelInput).fetch(PNetOutName(1)).run().get(0) //Index to get
    resultTwo.copyTo(PNetOutB)(0) //result.copyTo(Array.ofDim[Float](1, 512))(0)

    flipDiag(PNetOutP, PNetOutSizeW, PNetOutSizeH, 2)
    flipDiag(PNetOutB, PNetOutSizeW, PNetOutSizeH, 4)
    expand(PNetOutB, PNetOutBias)
    expandProb(PNetOutP, PNetOutProb)
    0

  }

  def updateBoxes(boxes: Vector[Box]): Vector[Box] = {
    val b = new Vector[Box]()
    for (i <- 0 until boxes.size()) {
      if (!boxes.get(i).deleted) b.addElement(boxes.get(i))
    }
    b
  }

  //Non-Maximum Suppression
  //Nms, the unqualified deleted is set to true
  private def nms(boxes: Vector[Box], threshold: Float, method: String): Unit = {
    //NMS. Pairwise comparison
    //int delete_cnt=0;
    val cnt = 0
    for (i <- 0 until boxes.size) {
      val box: Box = boxes.get(i)
      if (!box.deleted) { //score<0表示当前矩形框被删除
        for (j <- i + 1 until boxes.size) {
          val box2 = boxes.get(j)
          if (!box2.deleted) {
            val x1 = max(box.box(0), box2.box(0))
            val y1 = max(box.box(1), box2.box(1))
            val x2 = min(box.box(2), box2.box(2))
            val y2 = min(box.box(3), box2.box(3))
            breakable { //seun This might need to be above the if condition above this line
              if (x2 < x1 || y2 < y1) {
                break // seun there is no continue and break statement so this was used to replace continue
                val areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1)
                var iou = 0f
                if (method == "Union") iou = 1.0f * areaIoU / (box.area + box2.area - areaIoU)
                else if (method == "Min") {
                  iou = 1.0f * areaIoU / min(box.area, box2.area)
                  //Log.i(TAG, "[*]iou=" + iou) Seun commented this part out
                }
                if (iou >= threshold) { //删除prob小的那个框
                  if (box.score > box2.score) box2.deleted = true
                  else box.deleted = true
                  //delete_cnt++;
                }
              }
            }
          }
        }
      }
      //Log.i(TAG,"[*]sum:"+boxes.size()+" delete:"+delete_cnt);
    }
  }


  private def generateBoxes(prob: Array[Array[Float]], bias: Array[Array[Array[Float]]], scale: Float, threshold: Float, boxes: Vector[Box]) = {
    val h = prob.length
    val w = prob(0).length
    //Log.i(TAG,"[*]height:"+prob.length+" width:"+prob[0].length);
    for (y <- 0 until h) {
      for (x <- 0 until w) {
        val score = prob(y)(x)
        //only accept prob >threadshold(0.6 here)
        if (score > threshold) {
          val box = new Box()
          //score
          box.score = score
          //box
          box.box(0) = x * 2 / scale.round
          box.box(1) = y * 2 / scale.round
          box.box(2) = (x * 2 + 11) / scale.round
          box.box(3) = (y * 2 + 11) / scale.round
          //bbr
          for (i <- 0 until 4) {
            box.bbr(i) = bias(y)(x)(i)
          }
          //add
          boxes.addElement(box)
        }
      }
    }
    0
  }

  private def BoundingBoxReggression(boxes: Vector[Box]): Unit = {
    for(i <- 0 until boxes.size()){
      boxes.get(i).calibrate
    }
  }

  private def PNet(imageBuffer: BufferedImage, minSize: Int) = {
    val image = imageBuffer.asInstanceOf[BufferedImage]
    val whMin = min(image.getHeight(), image.getWidth())
    var currentFaceSize = minSize //currentFaceSize=minSize/(factor^k) k=0,1,2... until excced whMin
    val totalBoxes = new  Vector[Box]()
    //【1】Image Paramid and Feed to Pnet
    while ( {currentFaceSize <= whMin}) {
      val scale = 12.0f / currentFaceSize
      //(1)Image Resize
      val bm = imageResize(imageBuffer, scale)
      val w = bm.getWidth
      val h = bm.getHeight
      //(2)RUN CNN
      val PNetOutSizeW = (Math.ceil(w * 0.5 - 5) + 0.5).toInt
      val PNetOutSizeH = (Math.ceil(h * 0.5 - 5) + 0.5).toInt
      val PNetOutProb = Array.ofDim[Float](PNetOutSizeH, PNetOutSizeW)

      val PNetOutBias = Array.ofDim[Float](PNetOutSizeH, PNetOutSizeW, 4)
      pnetForward(bm, session = Session, PNetOutProb, PNetOutBias)
      //(3)数据解析
      val curBoxes = new Vector[Box]()
      generateBoxes(PNetOutProb, PNetOutBias, scale, PNetThreshold, curBoxes)
      //Log.i(TAG,"[*]CNN Output Box number:"+curBoxes.size()+" Scale:"+scale);
      //(4)nms 0.5
      nms(curBoxes, 0.5f, "Union")
      //(5)add to totalBoxes
      for (i <- 0 until curBoxes.size()){
        if (!curBoxes.get(i).deleted) totalBoxes.addElement(curBoxes.get(i))
      }
      //Face Size等比递增
      currentFaceSize /= factor
    }
    //NMS 0.7
    nms(totalBoxes, 0.7f, "Union")
    //BBR
    BoundingBoxReggression(totalBoxes)
    updateBoxes(totalBoxes)
  }

  def trial (): Float = {
    val arrayTrial = new Array[Float]()
  }
    object ScalaMtcnn {

      def main(args: Array[String]): Unit = {
        val mtcnn = new Mtcnn()
        val image = Paths.get("C:\\Users\\Seamfix\\Downloads\\ub.jpg")
        //mtcnn.convertBufferedImageToFloatBuffer(mtcnn.convertBase64ToBufferedImage(mtcnn.convertImageToBase64(image)))
        //mtcnn.pnetForward(mtcnn.convertBase64ToBufferedImage(mtcnn.convertImageToBase64(image)))
      }
    }
  }

