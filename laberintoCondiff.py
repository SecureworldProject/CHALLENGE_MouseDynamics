import pygame
import numpy as np
# Inicializar pygame
def laberinto():
    pygame.init()

    # Establecer dimensiones de la pantalla
    width = 640
    height = 640
    screen = pygame.display.set_mode((width, height))

    # Definir colores
    black = (0, 0, 0)
    white = (255, 255, 255)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    green=(0,255,0)
    #tiempo de captura de datos
    #------------------------------------------------------------------------------------
    intervalo=1
    datos=[]
    datosxy=[]
    #------------------------------------------------------------------------------------


    # Definir el laberinto
    maze = [[1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,0,0,0,1,1,1,0,2],
            [1,1,1,1,1,0,1,0,1,1,1,0,1],
            [1,0,0,0,0,0,1,0,1,0,0,0,1],
            [1,0,1,1,1,1,1,0,1,0,1,1,1],
            [1,0,0,0,1,0,0,0,1,0,1,1,1],
            [1,1,1,0,1,0,1,1,1,0,1,1,1],
            [3,0,1,0,1,0,0,0,1,0,1,1,1],
            [1,0,1,0,1,1,1,0,1,0,1,1,1],
            [1,0,1,0,1,1,0,0,1,0,0,0,1],
            [1,0,0,0,1,0,0,0,1,1,1,0,1],
            [1,1,1,1,1,0,1,1,1,1,1,0,1],
            [1,1,1,1,0,0,1,1,1,1,1,0,1],
            [1,0,0,0,0,1,1,0,0,0,0,0,1],
            [1,0,1,1,1,1,0,0,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,1,1,1,1,1], 
            [1,1,1,1,1,1,1,1,1,1,1,1,1]]
    columnas=width/len(maze[0])
    filas=height/len(maze)

    # Definir tamaño y posición del personaje y del final




    player_size = 15
    player_pos = [0*columnas, 7*filas+filas/2]

    # Dibujar el laberinto
    def draw_maze():
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                #si en el array hay un 1 es un muro
                if maze[i][j] == 1:
                    pygame.draw.rect(screen, black, (j*columnas, i*filas, columnas+1, filas+1))
                #si en el array hay un 2 es la salida del laberinto
                if maze[i][j] == 2:
                    pygame.draw.rect(screen, red, (j*columnas, i*filas, columnas+1, filas+1))
                #si en el array hay un 3 es la entrada del laberinto
                if maze[i][j] == 3:
                    pygame.draw.rect(screen, green, (j*columnas, i*filas, columnas+1, filas+1))
    # Dibujar el personaje 
    def draw_player_and_end():
        pygame.draw.circle(screen, blue, (player_pos[0]+player_size, player_pos[1]+player_size), player_size//2)

    # Verificar colisiones
    


    def check_collision(i,j):
        if maze[i][j]==1:
            return True
       
        elif maze[i][j]==2:
            return "WIN"
        else:
            return False




    # Loop principal del juego
    game_over = False
    relojPrincipal=pygame.time.Clock()
    #creamos una variable que nos impida movernos hasta que lleguemos al cuadrado de la salida
    mover=False
    #pintamos todos los elementos 
    screen.fill(white)
    draw_maze()
    draw_player_and_end()
    pygame.display.update()
    last=None
    while not game_over:
        # Manejo de eventos
        no_hay_evento=True
        relojPrincipal.tick(25) # 25 fps  (una muestra cada 40 ms)
        for event in pygame.event.get():
            no_hay_evento=False
            if check_collision(int((player_pos[1]+player_size)//filas),int((player_pos[0]+player_size)//columnas)) == "WIN":
                        print("¡Ganaste!")
                        game_over = True
            #acaba el programa si cerramos la ventana
            if event.type == pygame.QUIT:
                game_over = True
            #va a entrar en el bucle hasta que el raton del ordenador llegue al cuadrado de entrada al laberinto
            if mover==False:
                #coge la posicion del raton 
                mouse_pos = pygame.mouse.get_pos()
                #creamos el cuadrado de entrada al laberinto y comprobamos si el raton choca con el cuadrado
                wall_rect = pygame.Rect(0*columnas, 7*filas, columnas, filas)
                #comprobamos si la posicion del raton colisiona con el cuadrado de salida
                if wall_rect.collidepoint(mouse_pos):
                    #-------------------------------------------------------------------------------------------------
                    #guardamos el tiempo especifico para guardar nuestros datos
                    tiempo_intervalo=pygame.time.get_ticks()
                    #-------------------------------------------------------------------------------------------------
                    #cambiamos la variable mover para que se pueda mover el personaje
                    mover=True
                    last=mouse_pos
                
            else:
                #coge la posicion del raton 
                
                mouse_pos = pygame.mouse.get_pos()
                
                
                #target_pos = [mouse_pos[0]-player_size, mouse_pos[1]-player_size]

                target_pos = [mouse_pos[0]-player_size, mouse_pos[1]-player_size]
                
                # Actualizar posición del personaje
                if target_pos is not None:
                    #guardamos la posicion del personaje en una variable auxiliar 
                    aux=[player_pos[0],player_pos[1]]
                
                    #movimientos mas lentos 
                    

                    #comprobamos si hay colision, si hay colision con un muro no se actualiza la posicion del personaje
                    #si colisiona con el cuadrado de salida del laberinto acaba el programa
                    #y si no colisiona actualiza la posicion del personaje
                    x_diff = target_pos[0] - player_pos[0]
                    y_diff = target_pos[1] - player_pos[1]
                    

                    #control para no pegar saltos gigantes
                    if abs(x_diff)>columnas or abs(y_diff)>filas:
                        x_diff=0
                        y_diff=0
                    
                        

                    #control movimiento bola en x, en y o en ambas dimensiones

                    if check_collision(int((player_pos[1]+player_size)//filas),int((player_pos[0]-1+player_size/2)//columnas))== True and x_diff<=0:

                    
                        player_pos[0]=aux[0]
                    elif check_collision(int((player_pos[1]+player_size)//filas),int((player_pos[0]+player_size*1.5)//columnas))== True and x_diff>=0:
                        player_pos[0]=aux[0]
                    else:
                        if abs(x_diff)>0:
                            player_pos[0] += x_diff 

                    if check_collision(int((player_pos[1]+1+player_size/2)//filas),int((player_pos[0]+player_size)//columnas))== True and y_diff<=0:

                    
                        player_pos[1]=aux[1]
                    elif check_collision(int((player_pos[1]+player_size*1.5)//filas),int((player_pos[0]+player_size)//columnas))== True and y_diff>=0:

                    
                        player_pos[1]=aux[1]
                    else:
                        if abs(y_diff)>0:
                            player_pos[1] += y_diff
                #-----------------------------------------------------------------------------------------------------------------
                #si el tiempo actual - el tiempo de la ultima vez que se ha optenido un dato es igual o mayor que el intervalo captura
                #el dato y reinicia el tiempo """
                    if pygame.time.get_ticks()-tiempo_intervalo>=intervalo:
                                tiempo_intervalo=pygame.time.get_ticks()
                                if (last!=None):
                                    hipo=(mouse_pos[0]-last[0])**2 + (mouse_pos[1]-last[1])**2
                                #datos.append([mouse_pos[0],mouse_pos[1]])
                                    datos.append(hipo)
                                    deltax=mouse_pos[0]-last[0]
                                    deltay=mouse_pos[1]-last[1]
                                    #datosxy.append([deltax,deltay])
                                    datosxy.append([deltax,deltay,hipo])
                                    
                                    #print (datos[-1], datosxy[-1])
                                
                                
                #-----------------------------------------------------------------------------------------------------------------

                
                    
                    screen.fill(white)
        
                    draw_maze()
                    draw_player_and_end()


                # Dibujar elementos en pantalla
            
            pygame.display.update()
        """    
        if no_hay_evento and mover:
            tiempo_intervalo=pygame.time.get_ticks()
            #datos.append([mouse_pos[0],mouse_pos[1]])
            datos.append(0)
        """
        #print ([mouse_pos[0],mouse_pos[1]])
        #if (len(datos)>0):
        #    print (datos[-1])

        last=mouse_pos
        
            #print(tiempo_intervalo)
    # Cerrar Pygame
    pygame.quit()
    #return datos, datosxy
    return datosxy
#-------------------------------------------------------------------

#ver que hacer con los datos y que devuelva los datos
if __name__ == "__main__":
    import time
    Datos, Datosxy=laberinto()
    Datos.pop(0)
    Datosxy.pop(0)
    """
    print(len(Datos))
    print(Datos)
    print("===============================")
    print(len(Datosxy))
    print(Datosxy)
    """
    milis= round((time.time() * 1000)) %1000
    #np.save('datos2/1.npy', Datos)
    np.save("datos2/deltaxy/hipo"+str(milis)+".npy", Datos)
    np.save("datos2/deltaxy/deltaxy"+str(milis)+".npy", Datosxy)

    
