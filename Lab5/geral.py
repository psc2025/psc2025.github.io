import numpy as np 
import cv2
from tqdm import tqdm
import time
import os
from matplotlib import pyplot as plt

def calibragem(lab, pathL, pathR):
    print("Extracting image coordinates of respective 3D pattern ....\n")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    img_ptsL = []
    img_ptsR = []
    obj_pts = []

    for i in tqdm(range(1, 19)):
        imgL = cv2.imread(pathL + f"img{i}.png")
        imgR = cv2.imread(pathR + f"img{i}.png")

        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        outputL = imgL.copy()
        outputR = imgR.copy()

        retR, cornersR = cv2.findChessboardCorners(outputR, (8, 6), None)
        retL, cornersL = cv2.findChessboardCorners(outputL, (8, 6), None)

        if retR and retL:
            obj_pts.append(objp)
            cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(outputR, (8, 6), cornersR, retR)
            cv2.drawChessboardCorners(outputL, (8, 6), cornersL, retL)
            cv2.imshow('cornersR', outputR)
            cv2.imshow('cornersL', outputL)

            img_ptsL.append(cornersL)
            img_ptsR.append(cornersR)

            cv2.imwrite(f'/home/ufabc/Documentos/cv2025/{lab}/data/StereoL_Calibrated/img{i}.png', outputR)
            cv2.imwrite(f'/home/ufabc/Documentos/cv2025/{lab}/data/StereoR_Calibrated/img{i}.png', outputL)

    calcula_params(obj_pts, img_ptsL, img_ptsR, imgL_gray, imgR_gray)

def calcula_params(obj_pts, img_ptsL, img_ptsR, imgL_gray, imgR_gray):
    print("Calculating left camera parameters ... ")
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)
    hL, wL = imgL_gray.shape[:2]
    alpha = 0
    new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), alpha, (wL, hL))

    print("Calculating right camera parameters ... ")
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None)
    hR, wR = imgR_gray.shape[:2]
    new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), alpha, (wR, hR))

    print("Stereo calibration .....")
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
        obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], criteria_stereo, flags)

    rectify_scale = 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(
        new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], Rot, Trns, rectify_scale, (0, 0))

    Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, imgL_gray.shape[::-1], cv2.CV_16SC2)
    Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, imgR_gray.shape[::-1], cv2.CV_16SC2)

    print("Saving parameters ......")
    cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
    cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
    cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
    cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
    cv_file.release()

def tirar_fotos(pathL, PathR, lab):
    print("Checking the right and left camera IDs:")
    print("Press (y) if IDs are correct and (n) to swap the IDs")
    print("Press enter to start the process >> ")

    indices = detectar_cameras()
    if len(indices) < 2:
        print("Menos de duas câmeras detectadas. Conecte duas e tente novamente.")
        return

    CamL_id = indices[0]
    CamR_id = indices[1]

    CamL = cv2.VideoCapture(CamL_id)
    CamR = cv2.VideoCapture(CamR_id)

    for i in range(100):
        retL, frameL = CamL.read()
        retR, frameR = CamR.read()

    cv2.imshow('imgL', frameL)
    cv2.imshow('imgR', frameR)
    print("nada")
    CamR.release()
    CamL.release()

    CamL = cv2.VideoCapture(CamL_id)
    CamR = cv2.VideoCapture(CamR_id)

    start = time.time()
    T = 2
    count = 0

    while True:
        timer = T - int(time.time() - start)
        retR, frameR = CamR.read()
        retL, frameL = CamL.read()

        img1_temp = frameL.copy()
        cv2.putText(img1_temp, f"{timer}", (50, 50), 1, 5, (55, 0, 0), 5)
        cv2.imshow('imgR', frameR)
        cv2.imshow('imgL', img1_temp)

        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

        retR, cornersR = cv2.findChessboardCorners(grayR, (8, 6), None)
        retL, cornersL = cv2.findChessboardCorners(grayL, (8, 6), None)

        if (retR == True) and (retL == True) and timer <= 0 and count <20:
            count += 1
            cv2.imwrite(f'/home/ufabc/Documentos/cv2025/{lab}/data/StereoR_Calibrated/img{count}.png', frameR)
            cv2.imwrite(f'/home/ufabc/Documentos/cv2025/{lab}/data/StereoL_Calibrated/img{count}.png', frameL)

        if timer <= 0:
            start = time.time()

        if (cv2.waitKey(1) & 0xFF == 27) or (count>=20) :
            print("Closing the cameras!")
            break

    CamR.release()
    CamL.release()
    cv2.destroyAllWindows()
    calibragem(lab, pathL, pathR)

def tirar_fotos_especifico(lab):
    indices = detectar_cameras()

    if len(indices) < 2:
        print("Menos de duas câmeras detectadas. Conecte duas e tente novamente.")
        return

    CamL_id = indices[0]
    CamR_id = indices[1]

    CamL = cv2.VideoCapture(CamL_id)
    CamR = cv2.VideoCapture(CamR_id)

    # Espera para capturar a primeira imagem de cada câmera
    for i in range(100):
        retL, frameL = CamL.read()
        retR, frameR = CamR.read()

    cv2.imshow('imgL', frameL)
    cv2.imshow('imgR', frameR)
    
    print("nada")
    CamL.release()
    CamR.release()

    CamL = cv2.VideoCapture(CamL_id)
    CamR = cv2.VideoCapture(CamR_id)

    # Lê os parâmetros de calibração
    print("Reading parameters ......")
    cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    cv_file.release()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # ou use 'mp4v' para .mp4
    out = cv2.VideoWriter('Movie3D.avi', fourcc, 24.0, (700, 700)) 
    count = 0

    while True:
        retR, imgR = CamR.read()
        retL, imgL = CamL.read()
        cv2.imshow('imgR', imgR)
        cv2.imshow('imgL', imgL)

        if retL and retR:
            imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

            # Aplica o remapeamento para corrigir as distorções
            Left_nice = cv2.remap(imgL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            Right_nice = cv2.remap(imgR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

            count += 1

            k = cv2.waitKey(0)
            print('Aperte "s" para tirar fotos')

            if k & 0xFF == 27:  # Tecla ESC para sair
                print("Fechando as câmeras!")
                break

            elif k == ord('s'):  # Tecla "s" para salvar e continuar
                output = Right_nice.copy()
                output = cv2.resize(output, (700, 700))
                cv2.namedWindow("3D movie", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("3D movie", 700, 700)
                cv2.imshow("3D movie", output)
                cv2.imwrite(f'/home/ufabc/Documentos/cv2025/{lab}/data/objeto/img_obj_{count}.png', output)

        else:
            break

    out.release()
    cv2.destroyAllWindows()

def detectar_objeto_video():
    print("Carregando parâmetros de calibração...")
    # Carrega parâmetros de calibração
    cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    cv_file.release()

    # Carrega imagem de referência
    nome_query_image = str(input("Insira o nome do objeto a ser buscado (ex: img_obj_1.png): "))
    query_img = cv2.imread(f"data/objeto/{nome_query_image}", cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        print("Erro ao carregar imagem de referência!")
        return

    # Inicializa SIFT e detecta features da imagem de referência
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query_img, None)

    # Configura FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Captura de vídeo das câmeras estéreo
    indices = detectar_cameras()
    if len(indices) < 2:
        print("Menos de duas câmeras detectadas.")
        return

    CamL = cv2.VideoCapture(indices[0])
    CamR = cv2.VideoCapture(indices[1])

    print("Buscando objeto... Pressione ESC para cancelar manualmente.")

    count = 0

    while True:
        retL, frameL = CamL.read()
        retR, frameR = CamR.read()
        if not retL or not retR:
            print("Erro ao capturar frames.")
            break

        # Remapeia para remover distorções
        Left_nice = cv2.remap(frameL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4)
        Right_nice = cv2.remap(frameR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4)

        # Converte imagem da esquerda para escala de cinza
        scene_gray = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

        kp2, des2 = sift.detectAndCompute(scene_gray, None)
        if des2 is None or len(des2) < 2:
            continue  # pula frame se não houver descritores suficientes

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except cv2.error as e:
            print("Erro ao fazer knnMatch:", e)
            continue


        # Lowe ratio test com mais rigor
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:  # menos tolerante
                good.append(m)

        MIN_MATCH_COUNT = 12  # Mais exigente
        print('good',len(good))

        if len(good) >= MIN_MATCH_COUNT:

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = query_img.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Desenha a caixa no objeto detectado
                Left_nice = cv2.polylines(Left_nice, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                # Salva a imagem com detecção
                count += 1
                path = f"data/objeto/deteccao_{count}.png"
                cv2.imwrite(path, Left_nice)
                print(f"✅ Sucesso ao achar o objeto! Imagem salva em: {path}")

                # Mostra resultado rapidamente
                cv2.imshow("Detecção Final", Left_nice)
                cv2.waitKey(1000)
                break  # Encerra após detectar com sucesso

        # Mostra os frames ao vivo
        cv2.imshow('Left Camera - Object Detection', Left_nice)
        cv2.imshow('Right Camera', Right_nice)

        # ESC = sair manualmente
        if cv2.waitKey(1) & 0xFF == 27:
            print("Encerrado manualmente pelo usuário.")
            break

    CamL.release()
    CamR.release()
    cv2.destroyAllWindows()



def detectar_cameras(max_cameras=10):
    print("Procurando por câmeras conectadas...")
    indices_disponiveis = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            print(f"Câmera encontrada no índice {i}")
            indices_disponiveis.append(i)
        cap.release()
    return indices_disponiveis

def detectar_obj():
    
    MIN_MATCH_COUNT = 10
    nome_query_image = str(input("Insira o nome do objeto a ser buscado"))
    nome_image = str(input("Insira o nome da imagem a ser buscada"))
    
    img1 = cv2.imread(f"data/objeto/{nome_query_image}", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f"data/objeto/{nome_image}", cv2.IMREAD_GRAYSCALE)
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2)
    
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        
        plt.imshow(img3, 'gray'),plt.show()
 
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

def detectar_circulo(cv2, frame, ref_r):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, 1, gray.shape[0] / 64,
        param1=200, param2=20,
        minRadius=5, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            x, y, r = c
            # converte para int para evitar overflow no cálculo da diferença
            if abs(int(r) - int(ref_r)) < 5:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                return True, frame
    return False, frame


def detectar_circulo_especifico_video():
    # === Passo 1: Ler imagem de referência e detectar círculo ===
    modelo_path = input("Digite o nome da imagem com o círculo de referência: ")
    modelo = cv2.imread(f"data/objeto/{modelo_path}")
    modelo_gray = cv2.cvtColor(modelo, cv2.COLOR_BGR2GRAY)
    modelo_blur = cv2.medianBlur(modelo_gray, 5)

    modelo_circles = cv2.HoughCircles(modelo_blur, cv2.HOUGH_GRADIENT, 1,
                                       modelo.shape[0] / 64,
                                       param1=200, param2=20,
                                       minRadius=5, maxRadius=30)

    if modelo_circles is None:
        print("❌ Nenhum círculo encontrado na imagem de referência.")
        return

    modelo_circles = np.uint16(np.around(modelo_circles))
    ref_circle = modelo_circles[0, 0]  # Usa o primeiro círculo detectado
    ref_x, ref_y, ref_r = ref_circle
    print(f"🔍 Círculo de referência detectado: centro=({ref_x},{ref_y}), raio={ref_r}")

    # === Passo 2: Ler parâmetros e abrir câmeras ===
    cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    cv_file.release()

    indices = detectar_cameras()
    if len(indices) < 2:
        print("❌ Menos de duas câmeras disponíveis.")
        return

    CamL = cv2.VideoCapture(indices[0])
    CamR = cv2.VideoCapture(indices[1])
    count = 0

    print("🎥 Iniciando detecção do círculo específico. Pressione ESC para sair.")
    cv2.namedWindow("Câmeras Esquerda (esq) e Direita (dir)", cv2.WINDOW_NORMAL)

    while True:
        retL, frameL = CamL.read()
        retR, frameR = CamR.read()

        if not retL or not retR:
            print("Falha ao capturar frame da câmera")
            break

        # Remapeamento para correção
        Left_nice = cv2.remap(frameL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4)
        Right_nice = cv2.remap(frameR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4)

        matchL, Left_nice = detectar_circulo(cv2, Left_nice, ref_r)
        matchR, Right_nice = detectar_circulo(cv2, Right_nice, ref_r)

        # Combina as imagens lado a lado para exibir
        combined = np.hstack((Left_nice, Right_nice))
        cv2.imshow("Câmeras Esquerda (esq) e Direita (dir)", combined)

        key = cv2.waitKey(1) & 0xFF
        if matchL or matchR or key == 27:
            cv2.imwrite(f"data/objeto/circulo_encontrado_left_{count}.png", Left_nice)
            cv2.imwrite(f"data/objeto/circulo_encontrado_right_{count}.png", Right_nice)
            break

        if key == 27:  # ESC
            print("⛔ Encerrado pelo usuário.")
            break

    CamL.release()
    CamR.release()
    cv2.destroyAllWindows()

import cv2
import os
import time
import numpy as np

dataset_path = "dataset_faces"

def criar_diretorio_dataset():
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)


def capturar_fotos_rosto(nome_usuario, num_fotos=20):
    criar_diretorio_dataset()

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print(f"[INFO] Capturando {num_fotos} fotos do rosto de {nome_usuario}. Olhe para a câmera...")

    count = 0
    countdown_time = 5  # segundos para o countdown

    while count < num_fotos:
        start_time = None
        rosto_capturado = False

        while not rosto_capturado:
            ret, frame = cam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                if start_time is None:
                    start_time = time.time()

                elapsed = time.time() - start_time
                countdown = countdown_time - int(elapsed)

                (x, y, w, h) = faces[0]
                rosto = gray[y:y+h, x:x+w]
                rosto = cv2.resize(rosto, (200, 200))

                # Mostrar o rosto detectado
                cv2.imshow("Capturando rosto", rosto)
            else:
                # Sem rosto detectado, reseta o tempo
                start_time = None
                countdown = countdown_time
                # Mostra uma janela preta para rosto
                cv2.imshow("Capturando rosto", np.zeros((200, 200), dtype=np.uint8))

            # Texto e countdown na imagem de preview
            cv2.putText(frame, f"Foto {count+1}/{num_fotos}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Tirando foto em: {countdown}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Preview - Olhe para a câmera", frame)

            # Quando o countdown acabar e rosto detectado, tira foto
            if start_time is not None and elapsed >= countdown_time:
                foto_path = f"{dataset_path}/{nome_usuario}_{count}.png"
                cv2.imwrite(foto_path, rosto)
                print(f"[INFO] Foto {count+1} salva em {foto_path}")
                count += 1
                rosto_capturado = True
                time.sleep(0.5)  # pausa rápida para dar tempo da câmera "descansar"

            # ESC para sair
            if cv2.waitKey(1) & 0xFF == 27:
                print("[INFO] Captura interrompida pelo usuário.")
                cam.release()
                cv2.destroyAllWindows()
                return

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captura de fotos para {nome_usuario} concluída.")
    
def treinar_eigenfaces():
    """
    Carrega as imagens do dataset, treina o modelo Eigenfaces e retorna o reconhecedor treinado.
    """
    print("[INFO] Carregando dataset e treinando Eigenfaces...")

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for filename in os.listdir(dataset_path):
        if filename.endswith(".png"):
            path = os.path.join(dataset_path, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            nome_usuario = filename.rsplit('_', 1)[0]

            if nome_usuario not in label_map:
                label_map[nome_usuario] = current_label
                current_label += 1
            
            faces.append(img)
            labels.append(label_map[nome_usuario])

    if len(faces) == 0:
        print("[ERRO] Dataset vazio! Capture fotos antes de treinar.")
        return None, None

    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    print("[INFO] Treinamento concluído.")
    return recognizer, label_map

def reconhecer_rosto_eigenface(recognizer, label_map):
    """
    Captura uma imagem da webcam, detecta o rosto e tenta reconhecê-lo usando o modelo Eigenfaces.
    """
    if recognizer is None or label_map is None:
        print("[ERRO] Modelo não treinado!")
        return

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("[INFO] Posicione seu rosto na frente da câmera para reconhecimento.")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rosto = gray[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, (200, 200))

            label_id, confianca = recognizer.predict(rosto)

            nome_usuario = None
            for nome, idx in label_map.items():
                if idx == label_id:
                    nome_usuario = nome
                    break

            texto = f"{nome_usuario} - Confiança: {confianca:.2f}"
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Reconhecimento Facial (Eigenfaces)", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

    cam.release()
    cv2.destroyAllWindows()



def run():
    n_lab = str(input("Número do lab: "))
    lab = f"Lab{n_lab}"

    pathR = f"/home/ufabc/Documentos/cv2025/{lab}/data/StereoL_Calibrated/"
    pathL = f"/home/ufabc/Documentos/cv2025/{lab}/data/StereoR_Calibrated/"

    foto = input("Deseja tirar fotos? (s/n): ")
    if foto == "s":
        tirar_fotos(pathL, pathR, lab)

    foto_obj = input("Deseja tirar fotos de objeto? (s/n): ")
    if foto_obj == "s":
        tirar_fotos_especifico(lab)

    cal = input("Deseja calibrar? (s/n): ")
    if cal == "s":
        calibragem(lab, pathL, pathR)

    objeto = str(input("Deseja reconhecer objeto?"))
    if objeto == "s":
        print("!!ATENÇÃO câmera deve estar calibrada")
        detectar_obj()
    
    video = input("Detecção em tempo real com vídeo (Homography)? (s/n): ")
    if video == "s":
        detectar_objeto_video()
    
    video = input("Detecção em tempo real com vídeo (Hough Transform)? (s/n): ")
    if video == "s":
        detectar_circulo_especifico_video()
    
    face = input("face?")
    if face == "s":
        capturar_fotos_rosto("usuario1", num_fotos=20)
        recognizer, label_map = treinar_eigenfaces()
        reconhecer_rosto_eigenface(recognizer, label_map)

if __name__ == '__main__':
    run()