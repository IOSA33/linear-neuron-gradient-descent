package main

import (
	"fmt"
	"math"
)

const inputX = 1
const outputY = 1
const outputTrueY float32 = 5
const steps = 5
const learningRate = 0.1

var weight float32 = 2
var bias float32 = 0

func main() {
	for i := 0; i < steps; i++ {
		var Ypred = Ypred(inputX)
		fmt.Println("Loss ->", LError(Ypred))
		ansForW, ansForb := gradientForWb(Ypred)
		weight = updateWeight(ansForW)
		bias = updateBias(ansForb)
		fmt.Printf("Step %d: Loss = %.4f, Ypred = %.2f, W = %.2f, b = %.2f\n", i, LError(Ypred), Ypred, weight, bias)
	}
}

func Ypred(x float32) float32 {
	return weight*x + bias
}

func LError(Ypred float32) float64 {
	return math.Pow(float64(Ypred-outputTrueY), 2)
}

func gradientForWb(Ypred float32) (float32, float32) {
	return 2 * (Ypred - outputTrueY) * inputX, 2 * (Ypred - outputTrueY)
}

func updateWeight(ansForW float32) float32 {
	return weight - (learningRate * ansForW)
}

func updateBias(ansForb float32) float32 {
	return bias - (learningRate * ansForb)
}
