package com.example.linearregression

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    var tflite: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {

        var input: EditText? = null
        var output: TextView? = null
        var predictionButton: Button? = null

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        input = findViewById(R.id.inputValueEditText)
        output = findViewById(R.id.modelResultsTextView)
        predictionButton = findViewById(R.id.predictButton)

        try {
            tflite = Interpreter(loadModel())
        } catch (ex: Exception){
            ex.printStackTrace()
        }

        predictionButton.setOnClickListener(View.OnClickListener {
            val predicted_y = useModel(input.getText().toString())
            output.setText(java.lang.Float.toString(predicted_y))
        })
    }

    // Loads the model
    @Throws(IOException::class)
    private fun loadModel(): MappedByteBuffer{
        val fileDescriptor = this.assets.openFd("linearRegressionModel.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val start = fileDescriptor.startOffset
        val end = fileDescriptor.declaredLength
        return  fileChannel.map(FileChannel.MapMode.READ_ONLY, start, end)
    }

    private fun useModel(inputX : String) : Float {
        val input = FloatArray(1)
        input[0] = inputX.toFloat()

        val output = Array(1){
            FloatArray(1)
        }
        tflite!!.run(input,output)
        return  output[0][0]
    }
}