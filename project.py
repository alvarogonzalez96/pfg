import cv2
import dlib
import numpy as np
import sys

# pueden ocurrir problemas si se importa cv2 despues de inference y utils
import jetson.inference
import jetson.utils


def main():

	# dimensiones de la ventana Y de las imagenes a capturar por la camara
	WIDTH = 800
	HEIGHT = 600

	# lista de trackers de dlib
	trackers= []

	# flag para pasar otra vez el frame por net.Detect() y detectar objetos
	redo_detection = False

	# numero de frames tras los cuales hacer un refresh de detecciones
	contador_frames = 0
	FRAMES_DETECT = 50

	# contadores segun la direccion de la persona
	contador_yendo_abajo = 0
	contador_yendo_arriba = 0

	# el modelo a cargar por defecto, poner en consola "--network={algo}"
	# para sobreescribir a ssd-mobilenet-v2
	net = jetson.inference.detectNet('ssd-mobilenet-v2', sys.argv, 0.99)

	# camara de opencv
	#indica el video a procesar
	cap = cv2.VideoCapture( 'L10.mp4')

	if cap.isOpened():

		writer = None

		# bucle principal
		while True:

			# captura imagen
			ret, img = cap.read()

			# este ret es una variable que se pone a "True" 
			# si la imagen se ha cogido bien
			if ret:

				# suma el contador de frames capturados por la camara
				contador_frames += 1

				# dlib necesita las imagenes en rgb
				img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

				if writer is None:
					fourcc = cv2.VideoWriter_fourcc(*"MJPG")
					writer = cv2.VideoWriter('output.mp4', fourcc, 30, (img.shape[1], img.shape[0]), True)


				# zona del bucle para las detecciones
				if contador_frames % FRAMES_DETECT == 0 or redo_detection:

					# reseteo del flag de darle a la tecla R de "refresh"
					redo_detection = False

					# vaciado de la lista de trackers para empezar de cero
					trackers = [] 

					# convierte la imagen en formato opencv (BGR unit8) a RGBA float32
					img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)

					# pasa la imagen RGBA float32 a memoria CUDA y hace las detecciones
					# utilizando Detectnet
					img_cuda = jetson.utils.cudaFromNumpy(img_rgba)
					detections = net.Detect(img_cuda, 800, 600, 'none')

					# coge la informacion de todas las detecciones y calcula las
					# esquinas superior izquierda y la inferior derecha porque dlib 
					# asi lo requiere
					for detection in detections:

						# coge las coordenadas de los rectangulos de las detecciones y
						# se pasan a un tracker por cada deteccion realizada, luego
						# guarda un numero junto con el tracker para ver en que mitad de
						# la imagen esta el objeto y asi determinar despues la
						# direccion (-1 arriba, 1 abajo)

						rectangle = [ \
							int(detection.Center[0] - detection.Width / 2), \
							int(detection.Center[1] - detection.Height / 2), \
							int(detection.Center[0] + detection.Width / 2), \
							int(detection.Center[1] + detection.Height / 2) \
						]

						# lista con (1) el tracker y (2) el numero de posicion
						list_w_tracker = []

						# crea un tracker de dlib
						tracker = dlib.correlation_tracker()

						# se pasan las esquinas del rectangulo
						rect_dlib = dlib.rectangle( \
							rectangle[0], \
							rectangle[1], \
							rectangle[2], \
							rectangle[3] \
						)

						# comienza el trackeo
						tracker.start_track(img_rgb, rect_dlib)

						# se añade el tracker a la lista
						list_w_tracker.append(tracker)

						# se comprueba si la direccion esta arriba
						if (rectangle[3] + (rectangle[1] - rectangle[3]) / 2) <= (250):

							list_w_tracker.append(-1)

						# se comprueba si la direccion esta abajo
						elif (rectangle[3] + (rectangle[1] - rectangle[3]) / 2) >= (250):

							list_w_tracker.append(1)

						# lista con las listas de los trackers
						trackers.append(list_w_tracker)

				# zona de trackeo, actualizacion de los trackers de dlib
				else:

					# coge (1) el tracker y (2) el numero de posicion
					for list_w_tracker in trackers:

						# para updatear un tracker hay que pasarle la imagen en formato rgb
						list_w_tracker[0].update(img_rgb)
						# saca el objeto de posicion que devuelve el tracker
						pos = list_w_tracker[0].get_position()

						# esquina superior izquierda
						upperLeftCornerX = int(pos.left())
						upperLeftCornerY = int(pos.top())

						# esquina inferior derecha
						lowerRightCornerX = int(pos.right())
						lowerRightCornerY = int(pos.bottom())

						# se comprueba si (1) el objeto se pasa de la mitad y (2) si
						# el numero de antes indicaba que estaba en la otra mitad
						if (upperLeftCornerY + (lowerRightCornerY - upperLeftCornerY) / 2) <= (250) and list_w_tracker[1] == 1:

							# el objeto se ha movido hacia arriba, cambia a -1
							list_w_tracker[1] = -1

							# actualizacion de contadores
							contador_yendo_arriba = contador_yendo_arriba + 1

						elif (upperLeftCornerY + (lowerRightCornerY - upperLeftCornerY) / 2) >= (250) and list_w_tracker[1] == -1:

							# el objeto se ha movido hacia abajo, cambia a 1
							list_w_tracker[1] = 1

							# actualizacion de contadores
							contador_yendo_abajo = contador_yendo_abajo + 1

						# pintar el marco de deteccion
						cv2.rectangle( \

							# imagen sobre la que pintar
							img, \

							# coordenadas de la esquina superior izquierda
							(upperLeftCornerX, upperLeftCornerY), \

							# coordenadas de la esquina inferior derecha
							(lowerRightCornerX, lowerRightCornerY), \

							# color BGR
							(0, 255, 0), \

							# grosor de linea
							2 \
						)

				cv2.putText( \

					# imagen sobre la que pintar
					img, \

					# texto
					f'Dentro -> {contador_yendo_arriba}', \

					# posicion del texto
					(0, 30), \

					# fuente
					cv2.FONT_HERSHEY_SIMPLEX, \

					# tamaño letra
					1.25, \

					# color BGR
					(0, 255, 0), \

					# grosor linea
					1 \
				)

				cv2.putText( \
					img, \
					f'Fuera -> {contador_yendo_abajo}', \
					(0, 65), \
					cv2.FONT_HERSHEY_SIMPLEX, \
					1.25, \
					(0, 255, 0), \
					1 \
				)

				cv2.putText( \
					img, \
					f'--------------------------------------', \
					(0, 250), \
					cv2.FONT_HERSHEY_SIMPLEX, \
					1.25, \
					(0, 255, 0), \
					1 \
				)



			if writer is not None:
				writer.write(img)
				# muestra imagen
				cv2.imshow('sth...', img)	

			# escucha pulsacion de tecla
			keyCode = cv2.waitKey(1) & 0xFF

			# para salir del bucle pulsar 'q' o 'esc'
			if keyCode == 27 or keyCode == ord('q'): 
				break

			# para resetear los contadores pulsar 'a'
			elif keyCode == ord('a'):
				contador_yendo_arriba = 0
				contador_yendo_abajo = 0

			# para resetear detecciones de forma manual pulsar 'r'
			elif keyCode == ord('r'):
				redo_detection = True

		if writer is not None:
			writer.release()

		# cerrar todo
		cap.release()
		cv2.destroyAllWindows()

	else:

		print('Unable to open camera')


if __name__ == '__main__':
	main()
