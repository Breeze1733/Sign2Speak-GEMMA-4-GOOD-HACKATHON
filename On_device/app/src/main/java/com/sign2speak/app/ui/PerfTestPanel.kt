package com.sign2speak.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.sign2speak.app.MainViewModel

@Composable
fun PerfTestPanel(viewModel: MainViewModel) {
    Column(Modifier.width(225.dp).background(Color(0xDD121212))) {
        Text("自动计时：手进入 → 置信度≥50%", color = Color.White)
        Button(onClick = { viewModel.beginPerfTrial("") }) {
            Text("清空并开始新一轮测量")
        }
        Text("每词识别后移出双手，再进入打下一个词。", color = Color.White)
    }
}
