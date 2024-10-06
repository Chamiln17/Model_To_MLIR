module {
  func.func @LinearRegressionModel(%arg0: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "tosa.const"() <{value = dense<-0.190472484> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %1 = "tosa.const"() <{value = dense<0.55728209> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    %2 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 1>} : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %3 = tosa.matmul %2, %0 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %4 = tosa.reshape %3 {new_shape = array<i64: 1, 1>} : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
    %5 = tosa.add %4, %1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %5 : tensor<1x1xf32>
  }
}

