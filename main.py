import pygame
import random
import glob
import os
import neat

pygame.init()

WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DinoEvolution")
font = pygame.font.SysFont("Arial", 20)
tag_font = pygame.font.SysFont("Arial", 13)
clock = pygame.time.Clock()

GROUND_Y = 500
CACTUS_Y_OFFSET = 12

# --- Load road ---
road_img = pygame.image.load("sprites/road.png").convert_alpha()
road_width = road_img.get_width()
road_content_rect = road_img.get_bounding_rect()
road_y = GROUND_Y - road_content_rect.top

# --- Load cacti ---
cactus_imgs = [
    pygame.image.load(f"sprites/cactus/{i}.png").convert_alpha()
    for i in range(1, 7)
]

# --- Dynamically discover all dino skins ---
# Looks for any_name_run1.png / _run2.png / _jump.png triplets in sprites/dino/
SKINS = {}
for path in glob.glob("sprites/dino/*_run1.png"):
    base = os.path.basename(path)
    skin_name = base.replace("_run1.png", "")
    run1_path = f"sprites/dino/{skin_name}_run1.png"
    run2_path = f"sprites/dino/{skin_name}_run2.png"
    jump_path = f"sprites/dino/{skin_name}_jump.png"

    if os.path.exists(run2_path) and os.path.exists(jump_path):
        SKINS[skin_name] = {
            "run": [
                pygame.image.load(run1_path).convert_alpha(),
                pygame.image.load(run2_path).convert_alpha(),
            ],
            "jump": pygame.image.load(jump_path).convert_alpha(),
        }

SKIN_NAMES = sorted(SKINS.keys())
print(f"Loaded {len(SKIN_NAMES)} dino skins: {SKIN_NAMES}")

GEN_COUNT = 0


class Dino:
    def __init__(self, genome, net, skin_name):
        self.genome = genome
        self.net = net
        self.skin_name = skin_name
        self.sprites = SKINS[skin_name]

        self.rect = self.sprites["run"][0].get_rect()
        self.rect.x = 50
        self.rect.bottom = GROUND_Y+12

        self.anim_index = 0
        self.anim_timer = 0.0
        self.anim_interval = 0.12

        self.vel_y = 0.0
        self.is_jumping = False
        self.alive = True

        self.distance = 0.0       # raw distance traveled
        self.passed_cacti = 0     # cacti successfully avoided
        self.reward = 0.0         # shaped reward used as fitness

    def think(self, next_cactus, scroll_speed):
        if next_cactus is None:
            dist, width, height = WIDTH, 0, 0
        else:
            dist = next_cactus["rect"].x - self.rect.right
            width = next_cactus["rect"].width
            height = next_cactus["rect"].height

        inputs = (dist / WIDTH, width / 100, height / 100, scroll_speed / 22)
        output = self.net.activate(inputs)
        if output[0] > 0.5 and not self.is_jumping:
            self.is_jumping = True
            self.vel_y = -18

    def update_physics(self, dt, gravity):
        if self.is_jumping:
            self.rect.y += int(self.vel_y * (dt * 60))
            self.vel_y += gravity * (dt * 60)
            if self.rect.bottom >= GROUND_Y:
                self.rect.bottom = GROUND_Y
                self.is_jumping = False
                self.vel_y = 0

    def update_animation(self, dt, scroll_speed):
        if not self.is_jumping:
            self.anim_timer += dt
            interval = self.anim_interval * (6 / scroll_speed)
            if self.anim_timer >= interval:
                self.anim_timer = 0
                self.anim_index = (self.anim_index + 1) % len(self.sprites["run"])

    def current_image(self):
        if self.is_jumping:
            return self.sprites["jump"]
        return self.sprites["run"][self.anim_index]

    def update_reward(self, dt, scroll_speed):
        # --- Reward system ---
        # Small continuous reward for staying alive/moving (like distance-based score)
        self.distance += scroll_speed * dt
        self.reward = self.distance * 0.1

        # Bonus reward each time a cactus is fully passed
        self.reward += self.passed_cacti * 15

        self.genome.fitness = self.reward

    def draw(self, screen):
        screen.blit(self.current_image(), self.rect)

        # Score (reward) above the name — stacked directly over the dino's head
        score_text = tag_font.render(f"{int(self.reward)}", True, (0, 0, 0))
        name_text = tag_font.render(self.skin_name, True, (90, 90, 90))

        cx = self.rect.centerx
        score_y = self.rect.top - 32
        name_y = self.rect.top - 16

        screen.blit(score_text, (cx - score_text.get_width() // 2, score_y))
        screen.blit(name_text, (cx - name_text.get_width() // 2, name_y))


def eval_genomes(genomes, config):
    global GEN_COUNT
    GEN_COUNT += 1

    dinos = []
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        skin = SKIN_NAMES[i % len(SKIN_NAMES)]  # cycle through all available skins
        dinos.append(Dino(genome, net, skin))

    road_x1, road_x2 = 0.0, float(road_width)
    scroll_speed = 6.0
    max_speed = 22.0
    acceleration = 0.0008
    gravity = 0.9

    cacti = []
    spawn_timer = 0.0
    next_spawn_in = random.uniform(1.0, 2.2)

    max_time = 60.0
    elapsed = 0.0

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        elapsed += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        alive_dinos = [d for d in dinos if d.alive]
        if not alive_dinos or elapsed >= max_time:
            break

        if scroll_speed < max_speed:
            scroll_speed += acceleration * (dt * 60)
        move = scroll_speed * (dt * 60)

        road_x1 -= move
        road_x2 -= move
        if road_x1 <= -road_width:
            road_x1 = road_x2 + road_width
        if road_x2 <= -road_width:
            road_x2 = road_x1 + road_width

        spawn_timer += dt
        if spawn_timer >= next_spawn_in:
            spawn_timer = 0
            next_spawn_in = random.uniform(0.8, 2.0)
            next_spawn_in = max(0.5, next_spawn_in - (scroll_speed - 6) * 0.02)
            img = random.choice(cactus_imgs)
            rect = img.get_rect()
            rect.x = WIDTH
            rect.bottom = GROUND_Y + CACTUS_Y_OFFSET
            cacti.append({"img": img, "rect": rect, "counted": set()})

        for c in cacti[:]:
            c["rect"].x -= move
            if c["rect"].right < 0:
                cacti.remove(c)

        ahead = [c for c in cacti if c["rect"].right > 50]
        next_cactus = min(ahead, key=lambda c: c["rect"].x) if ahead else None

        for d in alive_dinos:
            d.think(next_cactus, scroll_speed)
            d.update_physics(dt, gravity)
            d.update_animation(dt, scroll_speed)

            hitbox = d.rect.inflate(-10, -10)
            for c in cacti:
                # Reward bonus once per dino per cactus, the moment it's cleared
                if c["rect"].right < d.rect.left and id(d) not in c["counted"]:
                    d.passed_cacti += 1
                    c["counted"].add(id(d))

                if hitbox.colliderect(c["rect"].inflate(-10, -10)):
                    d.alive = False
                    break

            d.update_reward(dt, scroll_speed)

        screen.fill((255, 255, 255))
        screen.blit(road_img, (road_x1, road_y))
        screen.blit(road_img, (road_x2, road_y))

        for c in cacti:
            screen.blit(c["img"], c["rect"])

        for d in alive_dinos:
            d.draw(screen)

        info = font.render(
            f"Gen {GEN_COUNT}   Alive {len(alive_dinos)}/{len(dinos)}   Speed {scroll_speed:.1f}",
            True, (50, 50, 50)
        )
        screen.blit(info, (20, 20))

        pygame.display.flip()


def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes, 1000)
    print("Best genome:\n", winner)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")
    run(config_path)