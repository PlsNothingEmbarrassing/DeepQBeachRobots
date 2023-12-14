    new_time_step = environment.step(action_step.action.numpy())

    new_observation = tf.expand_dims(new_time_step.observation, axis=0)