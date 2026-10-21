from .agents import LCRobot, SmallRubbish, Obstacle
def beach_portrayal(agent):
    if isinstance(agent, LCRobot):
        return lc_robot_portrayal(agent)
    if isinstance(agent, SmallRubbish):
        return s_rubbish_portrayal(agent)
    if isinstance(agent, Obstacle):
        return obstacle_portrayal(agent)
    return {}
    

def lc_robot_portrayal(lc_robot):
    if lc_robot is None:
        raise AssertionError
    return {"Shape": "arrowHead",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": lc_robot.x,
        "y": lc_robot.y,
        "scale": 2,
        "heading_x": -1 if lc_robot.isBusy else 1,
        "heading_y":0,
        # "r":4,
        "Color":"blue",

    }#

def s_rubbish_portrayal(s_rubbish):
    if s_rubbish is None:
        raise AssertionError
    return {"Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": s_rubbish.x,
        "y": s_rubbish.y,
        "scale": 2,        
        # "r":4,
        "Color": "red"}

def obstacle_portrayal(obstacle):
    if obstacle is None:
        raise AssertionError
    return {"Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": obstacle.x,
        "y": obstacle.y,
        "scale": 2,        
        # "r":4,
        "Color": "blue"}