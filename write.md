Regresión Logística Multiclase (Softmax)

El proyecto tuvo como objetivo implementar la regresión logística desde cero y compararla con la versión de scikit-learn, analizando los tres enfoques: binario, One-vs-All (OvA) y multinomial o Softmax. Todo el desarrollo se realizó en Python mediante Google Colab utilizando el conjunto de datos del vino, que contiene trece variables y tres clases. El trabajo fue realizado por Yudith Diana Chalco Cerezo y Sara Cristine Ocon Tovar.

En la versión binaria, el modelo aprende a clasificar entre dos categorías mediante una sola frontera de decisión. El enfoque OvA extiende este principio entrenando varios modelos binarios, uno por cada clase, para distinguir “una clase frente a todas las demás”. Aunque este método es simple y rápido, puede generar probabilidades incoherentes, ya que cada modelo trabaja de forma independiente.

La regresión logística multinomial o Softmax resuelve este problema considerando todas las clases al mismo tiempo. De esta forma, las probabilidades obtenidas están relacionadas entre sí y su suma total siempre es uno, lo que da interpretaciones más consistentes. Este modelo utiliza descenso de gradiente para ajustar todos los parámetros simultáneamente.

Durante la implementación se aplicaron técnicas de estabilidad numérica para evitar errores en los cálculos exponenciales, además de normalizar los datos y controlar la tasa de aprendizaje. Con esto se observó una convergencia estable del modelo, reflejada en la curva de log-verosimilitud.

En la comparación de resultados, tanto el modelo OvA como el Softmax lograron una exactitud aproximada del 98 %, coincidiendo también con la versión multinomial de scikit-learn. Las matrices de confusión mostraron que ambos métodos clasifican correctamente casi todas las observaciones, con pocos errores entre clases cercanas.

En conclusión, el enfoque Softmax ofrece una interpretación más coherente y probabilidades mejor calibradas que OvA, aunque ambos pueden rendir de forma similar en datos bien separados. El trabajo permitió entender en profundidad cómo se generaliza la regresión logística al caso multiclase y cómo controlar los aspectos numéricos que garantizan su correcta convergencia.
