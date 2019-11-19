package com.facedetect

//import libraries
import org.tensorflow.{Graph, Session}
import java.nio.FloatBuffer
import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, File, FileInputStream, ByteArrayOutputStream}

import javax.imageio.ImageIO
import java.nio.file.{Files, Path, Paths}
import java.awt.Graphics2D
import java.awt.Image

import sun.misc.BASE64Decoder
import java.util.Base64

import scala.util.Try


case class TensorflowObject(graph: Graph, sess: Session)

class Mtcnn{
  // parameter
  private val factor = 0.709f
  private val PNetThreshold = 0.6f
  private val RNetThreshold = 0.7f
  private val ONetThreshold = 0.7f



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
    println(fb)
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


}

object ScalaMtcnn {

  def main(args: Array[String]): Unit = {
    val mtcnn = new Mtcnn()
    val image = Paths.get("C:\\Users\\Seamfix\\Downloads\\ML.png")
    mtcnn.convertBufferedImageToFloatBuffer(mtcnn.convertBase64ToBufferedImage(mtcnn.convertImageToBase64(image)))
  }


}
