import pygame
import random
import numpy as np

class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.largeur_ecran = 600
        self.hauteur_ecran = 400
        self.taille_bloc = 10
        self.vitesse_serpent = 15
        self.fenetre = pygame.display.set_mode((self.largeur_ecran, self.hauteur_ecran))
        pygame.display.set_caption("Snake AI")
        self.horloge = pygame.time.Clock()

        # Définir les actions possibles : [0: Gauche, 1: Droite, 2: Haut, 3: Bas]
        self.actions = [0, 1, 2, 3]

        self.reset()

    def reset(self):
        """ Réinitialiser l'état du jeu """
        self.x = self.largeur_ecran / 2
        self.y = self.hauteur_ecran / 2
        self.x_changement = 0
        self.y_changement = 0
        self.serpent = [[self.x, self.y]]
        self.longueur_serpent = 1
        self.nourriture = self.generer_nourriture()
        self.score = 0
        return self.get_state()

    def generer_nourriture(self):
        """ Générer une nouvelle position de nourriture """
        return [round(random.randrange(0, self.largeur_ecran - self.taille_bloc) / 10.0) * 10.0,
                round(random.randrange(0, self.hauteur_ecran - self.taille_bloc) / 10.0) * 10.0]

    def get_state(self):
        """ Retourner un vecteur d'état représentant la situation actuelle """
        return np.array([self.x, self.y, self.nourriture[0], self.nourriture[1]])

    def step(self, action):
        """ Faire avancer le jeu d'un pas en fonction de l'action choisie """
        # Changer la direction en fonction de l'action
        if action == 0:  # Gauche
            self.x_changement = -self.taille_bloc
            self.y_changement = 0
        elif action == 1:  # Droite
            self.x_changement = self.taille_bloc
            self.y_changement = 0
        elif action == 2:  # Haut
            self.y_changement = -self.taille_bloc
            self.x_changement = 0
        elif action == 3:  # Bas
            self.y_changement = self.taille_bloc
            self.x_changement = 0

        # Mettre à jour la position du serpent
        self.x += self.x_changement
        self.y += self.y_changement
        self.serpent.append([self.x, self.y])

        if len(self.serpent) > self.longueur_serpent:
            del self.serpent[0]

        # Vérifier la collision avec les bords ou avec soi-même
        done = False
        if self.x >= self.largeur_ecran or self.x < 0 or self.y >= self.hauteur_ecran or self.y < 0:
            done = True
        for bloc in self.serpent[:-1]:
            if bloc == [self.x, self.y]:
                done = True

        # Gestion de la nourriture
        reward = 0
        if self.x == self.nourriture[0] and self.y == self.nourriture[1]:
            self.nourriture = self.generer_nourriture()
            self.longueur_serpent += 1
            reward = 10  # Récompense pour avoir mangé la nourriture
            self.score += 1

        # Faible pénalité pour chaque mouvement 
        reward -= 0.1

        return self.get_state(), reward, done

    def render(self):
        """ Dessiner l'état actuel du jeu """
        self.fenetre.fill((0, 0, 0))  # Fond noir
        # Dessiner la nourriture
        pygame.draw.rect(self.fenetre, (0, 255, 0), [self.nourriture[0], self.nourriture[1], self.taille_bloc, self.taille_bloc])
        # Dessiner le serpent
        for bloc in self.serpent:
            pygame.draw.rect(self.fenetre, (255, 255, 255), [bloc[0], bloc[1], self.taille_bloc, self.taille_bloc])

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
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        # Construction du modèle après initialisation des hyperparamètres
        self.model = self.build_model()

    def build_model(self):
        """Construit le modèle de réseau neuronal pour l'agent DQN"""
        model = Sequential()
        model.add(Dense(24, input_shape=(4,), activation='relu'))  # Correction de input_shape
        model.add(Dense(24, activation='relu'))
        model.add(Dense(4, activation='linear'))  # 4 sorties pour chaque action possible
        
        # Assure que la variable learning_rate est bien utilisée ici
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))  # Correction
        return model

    def remember(self, state, action, reward, next_state, done):
        """Enregistre une expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choisit une action selon epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return random.choice(self.env.actions)  # Exploration
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploitation

    def replay(self):
        """Entraîne le modèle sur un échantillon de la mémoire"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sélectionne un minibatch aléatoire
       
env = SnakeEnv()  # Remplace par ton environnement Snake
agent = DQN(env)

episodes = 1000 # Nombre d'épisodes d'entraînement
batch_size = 32  # Taille des lots pour l'entraînement

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])

    for time in range(500):  # Limite de temps pour chaque épisode
        action = agent.act(state)  # Choisir une action
        next_state, reward, done = env.step(action)  # Exécuter l'action
        reward = reward if not done else -10  # Pénalité en cas de défaite
        next_state = np.reshape(next_state, [1, 4])
        
        agent.remember(state, action, reward, next_state, done)  # Mémoriser la transition
        state = next_state
        
        if done:
            print(f"Episode {e}/{episodes} - Score : {time}")
            break
        
        if len(agent.memory) > batch_size:
            agent.replay()  # Entraîner le modèle avec un batch

import pygame  # Assurez-vous d'importer pygame

def play_snake(env, agent, episodes=1, render=True):
    """Fait jouer l'agent au jeu Snake."""
    for episode in range(episodes):
        state = env.reset()  # Réinitialiser l'environnement
        state = np.reshape(state, [1, 4])  # Ajuster la forme de l'état
        
        done = False
        total_reward = 0
        
        while not done:
            if render:
                env.render()  # Affiche le jeu à chaque étape
            
            # L'agent choisit une action
            action = agent.act(state)
            
            # Exécute cette action dans l'environnement
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            
            # Passe à l'état suivant
            state = next_state
            total_reward += reward
            
            # Si le jeu est terminé (le serpent est mort)
            if done:
                print(f"Episode: {episode+1}, Score: {total_reward}")
                break

    # Fermer Pygame correctement
    pygame.quit()  # Ajoutez cette ligne pour fermer Pygame


play_snake(env, agent, episodes=5, render=True)
