import pygame
import random
import numpy as np
import os

class SnakeEnv:
    def __init__(self,nombre_nourriture=10):
        pygame.init()
        self.largeur_ecran = 320
        self.hauteur_ecran = 240
        self.taille_bloc = 20
        self.vitesse_serpent = 20
        self.fenetre = pygame.display.set_mode((self.largeur_ecran, self.hauteur_ecran))
        pygame.display.set_caption("Snake AI")
        self.horloge = pygame.time.Clock()

        # Définir les actions possibles : [0: Gauche, 1: Droite, 2: Haut, 3: Bas]
        self.actions = [0, 1, 2, 3]
        # Number of food items in the environment
        self.nombre_nourriture = nombre_nourriture
        self.reset()

    def reset(self):
        """ Réinitialiser l'état du jeu """
        self.x = round(random.randrange(0, self.largeur_ecran - self.taille_bloc) / self.taille_bloc) * self.taille_bloc
        self.y = round(random.randrange(0, self.hauteur_ecran - self.taille_bloc) / self.taille_bloc) * self.taille_bloc
        self.x_changement = 0
        self.y_changement = 0
        self.serpent = [[self.x, self.y]]
        self.longueur_serpent = 1
        self.nourriture = self.generer_nourriture()
        self.score = 0
        return self.get_state()

    # def generer_nourriture(self):
    #     """ Générer une nouvelle position de nourriture """
    #     return [round(random.randrange(0, self.largeur_ecran - self.taille_bloc) / self.taille_bloc) * self.taille_bloc,
    #             round(random.randrange(0, self.hauteur_ecran - self.taille_bloc) / self.taille_bloc) * self.taille_bloc]
    def generer_nourriture(self):
        """ Generate positions for multiple food items """
        nourriture = []
        for _ in range(self.nombre_nourriture):
            x = round(random.randrange(0, self.largeur_ecran - self.taille_bloc) / self.taille_bloc) * self.taille_bloc
            y = round(random.randrange(0, self.hauteur_ecran - self.taille_bloc) / self.taille_bloc) * self.taille_bloc
            nourriture.append([x, y])
        return nourriture
    
    def get_state(self):
        """Return a normalized state vector representing the current situation."""
        normalized_x = self.x / self.largeur_ecran
        normalized_y = self.y / self.hauteur_ecran
        # normalized_food_x = self.nourriture[0] / self.largeur_ecran
        # normalized_food_y = self.nourriture[1] / self.hauteur_ecran

        # return np.array([normalized_x, normalized_y, normalized_food_x, normalized_food_y])
        # Normalize positions of all food items
        normalized_food_positions = []
        for food in self.nourriture:
            normalized_food_x = food[0] / self.largeur_ecran
            normalized_food_y = food[1] / self.hauteur_ecran
            normalized_food_positions.extend([normalized_food_x, normalized_food_y])

        return np.array([normalized_x, normalized_y] + normalized_food_positions)

    def step(self, action):
        """ Faire avancer le jeu d'un pas en fonction de l'action choisie """
        # Changer la direction en fonction de l'action
        if action == 0 and self.x_changement != self.taille_bloc:  # Gauche
            self.x_changement = -self.taille_bloc
            self.y_changement = 0
        elif action == 1 and self.x_changement != -self.taille_bloc:  # Droite
            self.x_changement = self.taille_bloc
            self.y_changement = 0
        elif action == 2 and self.y_changement != self.taille_bloc:  # Haut
            self.y_changement = -self.taille_bloc
            self.x_changement = 0
        elif action == 3 and self.y_changement != -self.taille_bloc:  # Bas
            self.y_changement = self.taille_bloc
            self.x_changement = 0

        # Mettre à jour la position du serpent
        self.x += self.x_changement
        self.y += self.y_changement
        self.serpent.append([self.x, self.y])

        if len(self.serpent) > self.longueur_serpent:
            del self.serpent[0]
        reward = 0
        # Vérifier la collision avec les bords ou avec soi-même
        done = False
        if self.x >= self.largeur_ecran or self.x < 0 or self.y >= self.hauteur_ecran or self.y < 0:
            reward -=20
            done = True
        for bloc in self.serpent[:-1]:
            if bloc == [self.x, self.y]:
                done = True

        # Gestion de la nourriture
        for food in self.nourriture[:]:
            if self.x == food[0] and self.y == food[1]:
                self.nourriture.remove(food)
                self.nourriture.append(self.generer_nourriture()[0])  # Add a new food item
                self.longueur_serpent += 1
                reward += 10  # Reward for eating food
                self.score += 1
        # if self.x == self.nourriture[0] and self.y == self.nourriture[1]:
        #     self.nourriture = self.generer_nourriture()
        #     self.longueur_serpent += 1
        #     reward += 10  # Récompense pour avoir mangé la nourriture
        #     self.score += 1

        # Faible pénalité pour chaque mouvement 
        reward -= 0.05

        return self.get_state(), reward, done

    def draw_text(self, text, pos):
        """Affiche le texte à l'écran."""
        font = pygame.font.SysFont("Arial", 20)  # Choisissez une police et une taille
        text_surface = font.render(text, True, (255, 255, 255))  # Texte en blanc
        self.fenetre.blit(text_surface, pos)

    def render(self, epsilon=0):
        """Draws the current state of the game."""
        self.fenetre.fill((0, 0, 0))  # Black background
        # Draw the food
        # pygame.draw.rect(self.fenetre, (0, 255, 0), [self.nourriture[0], self.nourriture[1], self.taille_bloc, self.taille_bloc])
        for food in self.nourriture:
            pygame.draw.rect(self.fenetre, (0, 255, 0), [food[0], food[1], self.taille_bloc, self.taille_bloc])

        # Draw the snake
        for bloc in self.serpent:
            pygame.draw.rect(self.fenetre, (255, 255, 255), [bloc[0], bloc[1], self.taille_bloc, self.taille_bloc])
        
        # Display epsilon in the top-left corner
        epsilon_text = f"Epsilon: {epsilon:.2f}"
        self.draw_text(epsilon_text, (10, 10))
        
        pygame.display.update()
        self.horloge.tick(self.vitesse_serpent)


import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)
        
        # Hyperparamètres de DQN
        self.gamma = 0.95  # Facteur de discount
        self.epsilon = 1.0  # Facteur d'exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.batch_size = 256
        # Determine input size based on environment's state
        # State size = 2 (snake's normalized position) + 2 * number of fruits
        self.state_size = 2 + 2 * env.nombre_nourriture

        # Number of actions (e.g., [0: Left, 1: Right, 2: Up, 3: Down])
        self.action_size = len(env.actions)
        # Construction du modèle après initialisation des hyperparamètres
        self.model = self.build_model()

    def get_epsilon(self):
        return self.epsilon
    
    
        
    def build_model(self):
        """Construit le modèle de réseau neuronal pour l'agent DQN"""
        model = Sequential()
        # model.add(Dense(64, input_shape=(4,), activation='relu'))  # Correction de input_shape
        model.add(Dense(64, input_shape=(self.state_size,), activation='relu'))  # Correction de input_shape

        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.state_size, activation='linear'))  # 4 sorties pour chaque action possible
        
        # Assure que la variable learning_rate est bien utilisée ici
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))  # Correction
        return model

    def remember(self, state, action, reward, next_state, done):
        """Enregistre une expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))

    # def act(self, state):
    #     """Choisit une action selon epsilon-greedy"""
    #     if np.random.rand() <= self.epsilon:
    #         return random.choice(self.env.actions)  # Exploration
    #     q_values = self.model.predict(state)
    #     return np.argmax(q_values[0])  # Exploitation
    def act(self, state):
        """Choose an action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.choice(self.env.actions)  # Exploration
        state = np.reshape(state, [1, self.state_size])  # Reshape for prediction
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploitation
    
    def replay(self):
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if len(self.memory) < self.batch_size:
            return

        # Sample a minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Extract data from minibatch
        states = np.vstack([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.vstack([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Predict Q(s, a) for all states and next_states
        q_values = self.model.predict(states)
        q_values_next = self.model.predict(next_states)

        # Update target Q-values
        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(q_values_next[i])

        # Train the model
        self.model.fit(states, q_values, epochs=1, verbose=0)





       
env = SnakeEnv()
agent = DQN(env)

episodes = -1  # Définissez à -1 pour un entraînement infini

def save_model_with_unique_name(model, base_filename="snake_dqn.h5"):
    """Sauvegarde le modèle avec un nom de fichier unique dans le même dossier que le fichier Python."""
    # Récupérer le chemin du répertoire contenant le fichier Python actuel
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Chemin absolu du script actuel
    
    filename = os.path.join(script_dir, base_filename)  # Créer le chemin complet pour le fichier
    counter = 1
    
    # Si un fichier avec le même nom existe déjà, ajouter un numéro au nom
    while os.path.exists(filename):
        filename = os.path.join(script_dir, f"{os.path.splitext(base_filename)[0]}_{counter}.h5")
        counter += 1

    # Sauvegarder le modèle au format HDF5 dans le chemin spécifié
    model.save(filename)
    print(f"Modèle sauvegardé sous '{filename}'.")

episode_number = 0
running = True

while running:
    if episodes != -1 and episode_number >= episodes:
        break

    state = env.reset()
    # state = np.reshape(state, [1, 4])  # Ajuster si la taille de l'état a changé
    state_size = 2 + 2 * env.nombre_nourriture  # Determine the correct state size
    state = np.reshape(state, [1, state_size])  # Reshape state based on state_size
        
    time_step = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        reward = reward if not done else -10
        # next_state = np.reshape(next_state, [1, 4])
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        # Vérification des événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Appui sur la touche 'q'
                    running = False
                    break

        if not running:
            break

        
        if time_step % 256 == 0 and len(agent.memory) > agent.batch_size:
            agent.replay()

        env.render(agent.get_epsilon())

        time_step += 1

    if not running:
        # Sauvegarde du modèle lors de l'interruption
        print("ON SAUVEGARDE LE MODELE!!!")
        save_model_with_unique_name(agent.model)
        break

    episode_number += 1
    print(f"Épisode {episode_number} terminé. Score: {time_step}")

    # Sauvegarde du modèle après chaque épisode (optionnel)
    # save_model_with_unique_name(agent.model)

# Sauvegarde du modèle à la fin de l'entraînement normal
if running:
    save_model_with_unique_name(agent.model)


import numpy as np
import pygame

def play_snake(env, agent, episodes=1, render=True):
    """Allows the agent to play the Snake game."""
    for episode in range(episodes):
        state = env.reset()
        # state = np.reshape(state, [1, 4])
        state_size = 2 + 2 * env.nombre_nourriture  # Determine the correct state size
        state = np.reshape(state, [1, state_size])  # Reshape state based on state_size
        
        done = False
        total_reward = 0
        
        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                env.render(agent.get_epsilon())  # Pass epsilon value here
            
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # next_state = np.reshape(next_state, [1, 4])
            next_state = np.reshape(next_state, [1, state_size])

            
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {episode + 1}, Score: {total_reward}")
                break
    
    pygame.quit()

# Assurez-vous de démarrer Pygame avant de jouer
pygame.init()  # Initialiser Pygame
play_snake(env, agent, episodes=20, render=True)
