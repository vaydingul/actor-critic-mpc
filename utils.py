from gymnasium import spaces
import pygame

import numpy as np


def make_observation_space(size):
    return spaces.Dict(
        {
            "agent_location": spaces.Box(0, size, shape=(2,), dtype=float),
            "agent_velocity": spaces.Box(-0.001, 0.001, shape=(2,), dtype=float),
            "target_location": spaces.Box(0, size, shape=(2,), dtype=float),
            "target_velocity": spaces.Box(-0.001, 0.001, shape=(2,), dtype=float),
        }
    )


def make_action_space():
    return spaces.Box(-10.0, 10.0, shape=(2,), dtype=float)


def render_frame(
    size,
    window_size,
    agent_location_original=None,
    agent_velocity_original=None,
    target_location_original=None,
    target_velocity_original=None,
    action=None,
    distance_threshold=None,
    system=None,
    agent_location_noisy=None,
    agent_velocity_noisy=None,
    target_location_noisy=None,
    target_velocity_noisy=None,
    predicted_agent_location=None,
    predicted_agent_velocity=None,
    predicted_target_location=None,
    predicted_target_velocity=None,
):
    TEXT_SIZE = 10

    canvas = pygame.Surface((window_size, window_size), pygame.SRCALPHA)
    canvas.fill((255, 255, 255))

    distance = np.linalg.norm(agent_location_original - target_location_original, ord=2)

    # Draw the agent location.
    agent_location_original = scale_vector(agent_location_original, size, window_size)
    agent_size = scale_size(0.4, size, window_size)
    pygame.draw.circle(
        canvas, (255, 0, 0), agent_location_original, agent_size, width=0
    )

    # Draw the agent location noisy if it is provided. Shaded red
    if agent_location_noisy is not None:
        agent_location_noisy = scale_vector(agent_location_noisy, size, window_size)
        agent_size = scale_size(0.4, size, window_size)
        pygame.draw.circle(
            canvas, (255, 0, 0, 100), agent_location_noisy, agent_size, width=0
        )

    # Put text AGENT on top of the agent.
    text = pygame.font.SysFont("Helvetica", 20).render("A", True, (255, 255, 255))
    text_rect = text.get_rect()
    text_rect.center = agent_location_original
    canvas.blit(text, text_rect)

    # Draw the agent velocity
    agent_velocity_original = scale_vector(agent_velocity_original, size, window_size)
    pygame.draw.line(
        canvas,
        (255, 0, 0),
        agent_location_original,
        (
            agent_location_original[0] + agent_velocity_original[0],
            agent_location_original[1] + agent_velocity_original[1],
        ),
        width=2,
    )

    # Draw the agent velocity noisy if it is provided. Shaded red
    if agent_velocity_noisy is not None:
        agent_velocity_noisy = scale_vector(agent_velocity_noisy, size, window_size)
        pygame.draw.line(
            canvas,
            (255, 0, 0, 10),
            agent_location_noisy,
            (
                agent_location_noisy[0] + agent_velocity_noisy[0],
                agent_location_noisy[1] + agent_velocity_noisy[1],
            ),
            width=2,
        )

    # Draw the predicted agent location if it is provided. Shaded red, smaller
    if predicted_agent_location is not None:
        for k in range(predicted_agent_location.shape[0]):
            predicted_agent_location_original = scale_vector(
                predicted_agent_location[k], size, window_size
            )
            predicted_agent_size = scale_size(0.2, size, window_size)
            pygame.draw.circle(
                canvas,
                (0, 0, 0, 100),
                predicted_agent_location_original,
                predicted_agent_size,
                width=0,
            )

    # Draw the target location.
    target_location_original = scale_vector(target_location_original, size, window_size)
    target_size = scale_size(0.4, size, window_size)
    pygame.draw.circle(
        canvas, (0, 0, 255), target_location_original, target_size, width=0
    )

    # Draw the target location noisy if it is provided. Shaded blue
    if target_location_noisy is not None:
        target_location_noisy = scale_vector(target_location_noisy, size, window_size)
        target_size = scale_size(0.4, size, window_size)
        pygame.draw.circle(
            canvas, (0, 0, 255, 10), target_location_noisy, target_size, width=0
        )

    # Put text TARGET on top of the target.
    text = pygame.font.SysFont("Helvetica", 20).render("T", True, (255, 255, 255))
    text_rect = text.get_rect()
    text_rect.center = target_location_original
    canvas.blit(text, text_rect)

    # Draw the target velocity
    target_velocity_original = scale_vector(target_velocity_original, size, window_size)
    pygame.draw.line(
        canvas,
        (0, 0, 255),
        target_location_original,
        (
            target_location_original[0] + target_velocity_original[0],
            target_location_original[1] + target_velocity_original[1],
        ),
        width=2,
    )

    # Draw the target velocity noisy if it is provided. Shaded blue
    if target_velocity_noisy is not None:
        target_velocity_noisy = scale_vector(target_velocity_noisy, size, window_size)
        pygame.draw.line(
            canvas,
            (0, 0, 255, 100),
            target_location_original,
            (
                target_location_original[0] + target_velocity_noisy[0],
                target_location_original[1] + target_velocity_noisy[1],
            ),
            width=2,
        )

    # Draw the predicted target location if it is provided. Shaded blue, smaller
    if predicted_target_location is not None:
        for k in range(predicted_target_location.shape[0]):
            predicted_target_location_original = scale_vector(
                predicted_target_location[k], size, window_size
            )
            predicted_target_size = scale_size(0.1, size, window_size)
            pygame.draw.circle(
                canvas,
                (0, 0, 0, 100),
                predicted_target_location_original,
                predicted_target_size,
                width=0,
            )

    # Draw a circle to indicate the distance threshold.
    distance_threshold = scale_size(distance_threshold, size, window_size)
    pygame.draw.circle(
        canvas, (0, 0, 0), target_location_original, distance_threshold, width=1
    )

    # Draw the distance to the target.
    pygame.draw.line(
        canvas,
        (0, 0, 0),
        agent_location_original,
        (target_location_original[0], target_location_original[1]),
        width=2,
    )
    # Put the distance in the middle of the line.
    text = pygame.font.SysFont("Helvetica", TEXT_SIZE).render(
        f"{distance:.2f}", True, (0, 0, 0)
    )

    text_rect = text.get_rect()
    text_rect.center = (
        (agent_location_original[0] + target_location_original[0]) / 2,
        (agent_location_original[1] + target_location_original[1]) / 2,
    )
    canvas.blit(text, text_rect)

    # Draw the action vector. Cyan
    action_vector = scale_vector(0.2 * action, size, window_size)
    pygame.draw.line(
        canvas,
        (0, 255, 255),
        agent_location_original,
        (
            agent_location_original[0] + action_vector[0] * 5,
            agent_location_original[1] + action_vector[1] * 5,
        ),
        width=4,
    )

    # Put additional information on the screen.
    # Action
    text = pygame.font.SysFont("Helvetica", TEXT_SIZE).render(
        f"Action: {action}", True, (0, 0, 0)
    )
    canvas.blit(text, (10, 10))

    # Agent location
    text = pygame.font.SysFont("Helvetica", TEXT_SIZE).render(
        f"Agent location: {agent_location_original}", True, (0, 0, 0)
    )
    canvas.blit(text, (10, 30))

    # Agent velocity
    text = pygame.font.SysFont("Helvetica", TEXT_SIZE).render(
        f"Agent velocity: {agent_velocity_original}", True, (0, 0, 0)
    )
    canvas.blit(text, (10, 50))

    # Target location
    text = pygame.font.SysFont("Helvetica", TEXT_SIZE).render(
        f"Target location: {target_location_original}", True, (0, 0, 0)
    )
    canvas.blit(text, (10, 70))

    # Target velocity
    text = pygame.font.SysFont("Helvetica", TEXT_SIZE).render(
        f"Target velocity: {target_velocity_original}", True, (0, 0, 0)
    )
    canvas.blit(text, (10, 90))

    # Distance
    text = pygame.font.SysFont("Helvetica", TEXT_SIZE).render(
        f"Distance: {distance}", True, (0, 0, 0)
    )
    canvas.blit(text, (10, 110))

    # If system has a wind gust, draw it.
    # Wind gust effective region is denominated by a variable called wind_gust_region.
    # wind_gust_region = [[x_min, x_max], [y_min, y_max]]
    # Values are in [0, 1]
    if system.wind_gust is not None:
        wind_gust_region_x_lower = system.wind_gust_region_x_lower
        wind_gust_region_x_upper = system.wind_gust_region_x_upper
        wind_gust_region_y_lower = system.wind_gust_region_y_lower
        wind_gust_region_y_upper = system.wind_gust_region_y_upper

        # convert to (left, top, width, height) format
        wind_gust_region_rect = (
            wind_gust_region_x_lower,
            wind_gust_region_y_lower,
            wind_gust_region_x_upper - wind_gust_region_x_lower,
            wind_gust_region_y_upper - wind_gust_region_y_lower,
        )

        wind_gust_region_rect = scale_rect(wind_gust_region_rect, size, window_size)

        pygame.draw.rect(canvas, (0, 255, 0), wind_gust_region_rect, width=1)
        # Write amount of wind_gust to the center of rectangle
        text = pygame.font.SysFont("Helvetica", TEXT_SIZE).render(
            f"{system.wind_gust}", True, (0, 0, 0)
        )
        text_rect = text.get_rect()
        text_rect.center = (
            wind_gust_region_rect[0] + wind_gust_region_rect[2] / 2,
            wind_gust_region_rect[1] + wind_gust_region_rect[3] / 2,
        )
        
        canvas.blit(text, text_rect)
    return canvas


def scale_rect(rect, size, window_size):
    return (
        int(rect[0] / size * window_size),
        int(rect[1] / size * window_size),
        int(rect[2] / size * window_size),
        int(rect[3] / size * window_size),
    )


def scale_vector(vector, size, window_size):
    return (
        int(vector[0] / size * window_size),
        int(vector[1] / size * window_size),
    )


def scale_size(size_, size, window_size):
    return int(size_ / size * window_size)
