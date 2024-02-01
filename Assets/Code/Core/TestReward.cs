/******************************************************************************
 * RObotic MAnipulation Network (ROMAN)                                       *
 * Hybrid Hierarchical Learning for Solving Complex Sequential Tasks          *
 * -------------------------------------------------------------------------- *
 * Purpose: The Main Code for the Expert Responsible for Pulling Objects      *
 ******************************************************************************/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.IO;


public class TestReward : Agent
{
    [Header("Master Network Controlled")]
    public bool agentControlled = false;
    public float agentWeight = 1.0f;

    [Header("Agent Settings")]
    public int stepSafetySync = 320;
    public float speed = 1f;

    [Header("Object Assignments")]
    public SceneManagement sceneManager;
    public Transform agentTR;
    public Transform drawerhandleTR;
    // +++ 3
    public Transform boxTopTargetTR;
    public Transform unpackingLocationTR;
    public ObjectInCollisionWith unpackLocationInCollisionWithBoxTopTarget;

    [Header("Low Pass Filter Settings")]
    public bool applyFilter = false;
    public float specifiedCutOff;
    public int specifiedSampleRate;
    public float specifiedResonance = Mathf.Sqrt(2);
    private const int requiredFilteredValues = 25;
    LowPassFilter[] lowPassFilter;

    [System.NonSerialized]
    public int episode = -1;
    [System.NonSerialized]
    EnvironmentNoise environmentNoise;

    private float actualTime;
    private Rigidbody agentRB;
    private GripperState gripperState;
    private ForceReading forceFingerA, forceFingerB;
    private Vector3 agentDefaultInitPos;

    void Awake()
    {
        agentDefaultInitPos = agentTR.localPosition;
    }

    // Initialise Agent Settings and Variables Thereafter Used
    void Start()
    {
        if (!GetComponent<EnvironmentNoise>())
            environmentNoise = gameObject.AddComponent<EnvironmentNoise>();
        else
            environmentNoise = gameObject.GetComponent<EnvironmentNoise>();

        agentRB = agentTR.GetComponent<Rigidbody>();
        gripperState = agentTR.GetComponent<GripperState>();
        forceFingerA = gripperState.RBFingerA.transform.GetComponent<ForceReading>();
        forceFingerB = gripperState.RBFingerB.transform.GetComponent<ForceReading>();

        lowPassFilter = new LowPassFilter[requiredFilteredValues];
        for (int i = 0; i < lowPassFilter.Length; i++)
        {
            lowPassFilter[i] = new LowPassFilter();
            lowPassFilter[i].ApplySettings(specifiedCutOff, specifiedSampleRate, specifiedResonance);
        }
    }

    // For New Episode
    public override void OnEpisodeBegin()
    {
        if (!agentControlled)
        {
            episodeHasResetForButtonPress = true;
            ResetEpisodeParameters();
            RandomizeEpisodeVariables();
        }
        episode++;
    }

    // Objects of Interest (OIs)
    private Vector3 handleTargetPosWithNoise;
    // +++ 
    private Vector3 boxTargetPosWithNoise, unpackLocationPosWithNoise;

    // State Space
    public override void CollectObservations(VectorSensor sensor)
    {
        // Input: Robot Proprioception (6)
        sensor.AddObservation(agentTR.localPosition);
        sensor.AddObservation(agentRB.velocity);

        // Input: Robot Force Sensor (2)
        sensor.AddObservation(forceFingerA.fNorm);
        sensor.AddObservation(forceFingerB.fNorm);

        // Input: Noisy Environment Data (3)
        handleTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(drawerhandleTR.localPosition, environmentNoise.posNoiseMeters);
        handleTargetPosWithNoise = LowPassFilteredPosition(handleTargetPosWithNoise, new int[] { 0, 1, 2 });
        sensor.AddObservation(handleTargetPosWithNoise);
        // Input: Noisy Environment Data (6) // +++ 6
        boxTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(boxTopTargetTR.localPosition, environmentNoise.posNoiseMeters);
        boxTargetPosWithNoise = LowPassFilteredPosition(boxTargetPosWithNoise, new int[] { 0, 1, 2 });
        sensor.AddObservation(boxTargetPosWithNoise);
        unpackLocationPosWithNoise = GaussianDistribution.PositionWGaussianNoise(unpackingLocationTR.localPosition, environmentNoise.posNoiseMeters);
        unpackLocationPosWithNoise = LowPassFilteredPosition(unpackLocationPosWithNoise, new int[] { 3, 4, 5 });
        sensor.AddObservation(unpackLocationPosWithNoise);
        sensor.AddObservation(distanceAgentToInitPosNorm);
    }

    // The Max Allowable Divergence After Which the Episode Terminates
    private float maxAllowableDivergence = 1.5f;

    // Action Space
    //public override void OnActionReceived(float[] vectorAction)
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Robot Position Control via Applied Force
        Vector3 controlSignalPos = Vector3.zero;
        controlSignalPos.y = actions.ContinuousActions[0] * agentWeight;
        controlSignalPos.z = actions.ContinuousActions[1] * agentWeight;
        controlSignalPos.x = actions.ContinuousActions[2] * agentWeight;
        Vector3 totalForcePos = controlSignalPos * speed * agentRB.mass;
        agentRB.AddForce(totalForcePos);

        // Open or Close the Gripper
        if (actions.ContinuousActions[3] * agentWeight < -0.9f)
            gripperState.openGripper = false;
        if (actions.ContinuousActions[3] * agentWeight > 0.9f)
            gripperState.openGripper = true;

        // Apply Agent-Specific Rewards   // +++ 1 what is the StepCount
        if (StepCount >= (MaxStep / (MaxStep <= 80000 ? 100 : stepSafetySync)))
            AgentSpecificReward();

        // Terminate if Agent Divergence is Higher than Threshold
        if (Vector3.Distance(agentDefaultInitPos, agentTR.localPosition) > maxAllowableDivergence)
            EndEpisode();
        // +++ 0 the function where called
        totalReward = GetCumulativeReward();
        PrintEpisodeInfo(false);
    }

    // +++ 3
    enum TaskState { Pull, PickDrop, PickPlace};
    TaskState currentTask = TaskState.Pull;

    void UpdateTaskState() {
        switch (currentTask) {
            case TaskState.Pull:
                if (drawerOpenDistanceNorm >= 0.95f) {// Drawer fully opened
                    currentTask = TaskState.PickDrop;
                    Debug.Log("Task-PickDrop:" + currentTask);
                }
                break;
            case TaskState.PickDrop:
                if (drawerOpenDistanceNorm < 0.90f) {// Drawer not fully opened
                    currentTask = TaskState.Pull;
                    Debug.Log("Task-Pull:" + currentTask);
                }
                // Criteria to switch to dropping, e.g., object is picked
                break;
            // Add other transitions as needed
        }
    }

    // Agent Specific Rewards
    public float totalReward = 0;
    //  +++ 1
    public bool boxTopInCollisionWithUnpackLocation = false;
    public float distanceAgentToHandleNorm, drawerOpenDistanceNorm, distanceAgentToInitPosNorm;
    // +++ 1
    public float distanceAgentToBoxTopTargetNorm, distanceBoxTopTargetToUnpackLocationNorm;

    public float maxDistanceAgentToHandle = -1f, maxDistanceAgentToInitPos = -1f;
    // +++ 1 question: max is set, but it can change during training
    public float maxDistanceAgentToBoxTopTarget = -1f, maxDistanceBoxTopTargetToUnpackLocation = -1f;
    [System.NonSerialized]
    public float distanceToBase = 0.0f;

    // +++
    void AgentSpecificReward()
    {
        float drawerOpenDistance = Vector3.Distance(handleTargetPosWithNoise, sceneManager.drawerHandleInitPos);
        drawerOpenDistanceNorm = Mathf.Clamp(drawerOpenDistance / (sceneManager.drawerCJ.linearLimit.limit * 2f), 0, 1);
        Debug.Log("drawerOpenDistanceNorm>=0.95?:" + drawerOpenDistanceNorm);
        Debug.Log("real-time-currentTask:" + currentTask);
        UpdateTaskState();

        if (currentTask == TaskState.Pull)
        {
            AddReward(CalculatePullingReward());
        }
        else if (currentTask == TaskState.PickDrop)
        {
            AddReward(CalculatePickingAndDroppingReward());
        }

        // +++ 1
        float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
        // Primary Goal is Achieved and Agent is Wihin Initial Position
        if (distanceAgentToInitPos < 0.1f && drawerOpenDistanceNorm >= 0.95f && boxTopInCollisionWithUnpackLocation && (StepCount >= (MaxStep / 10)) && !agentControlled)
        {
            SetReward(1000f);
            PrintEpisodeInfo(true);
            EndEpisode();
        }

    }


    float CalculatePullingReward()
    {
        // R1: Distance to Drawer Handle
        float distanceAgentToHandle = Vector3.Distance(agentTR.localPosition, handleTargetPosWithNoise);
        if (maxDistanceAgentToHandle < 0)
            maxDistanceAgentToHandle = distanceAgentToHandle;
        distanceAgentToHandleNorm = Mathf.Clamp(distanceAgentToHandle / maxDistanceAgentToHandle, 0, 1);
        float R1 = (1 - distanceAgentToHandleNorm) * 0.0005f * speed * (1 - drawerOpenDistanceNorm);

        // R2: Drawer Open Distance
        float drawerOpenDistance = Vector3.Distance(handleTargetPosWithNoise, sceneManager.drawerHandleInitPos);
        drawerOpenDistanceNorm = Mathf.Clamp(drawerOpenDistance / (sceneManager.drawerCJ.linearLimit.limit * 2f), 0, 1);
        float R2 = drawerOpenDistanceNorm * 0.005f * speed;

        // R3: Force Sensor Based Reward
        float R3 = 0f;
        if (!gripperState.openGripper)
            R3 = (forceFingerA.fNorm + forceFingerB.fNorm) * 0.0005f * speed * (1 - drawerOpenDistanceNorm);

        // R4: Distance to Initial Position
        float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
        if (maxDistanceAgentToInitPos < 0)
            maxDistanceAgentToInitPos = distanceAgentToInitPos;
        distanceAgentToInitPosNorm = Mathf.Clamp(distanceAgentToInitPos / maxDistanceAgentToInitPos, 0, 1);
        float R4 = (1 - distanceAgentToInitPosNorm) * 0.0005f * speed * drawerOpenDistanceNorm;
        distanceToBase = distanceAgentToInitPos;

        // Implement the logic from your first AgentSpecificReward method
        float pullingReward = R1 + R2 + R3 + R4;

        return pullingReward; // Calculate and return the pulling-specific reward
    }

    float CalculatePickingAndDroppingReward()
    {
        // R1: Agent Distance to Box Cover
        float distanceAgentToBoxTopTarget = Vector3.Distance(agentTR.localPosition, boxTargetPosWithNoise);
        if (maxDistanceAgentToBoxTopTarget < 0)
            maxDistanceAgentToBoxTopTarget = distanceAgentToBoxTopTarget;
        distanceAgentToBoxTopTargetNorm = Mathf.Clamp(distanceAgentToBoxTopTarget / maxDistanceAgentToBoxTopTarget, 0, 1);
        float R1 = (1 - distanceAgentToBoxTopTargetNorm) * 0.0005f * speed * (!boxTopInCollisionWithUnpackLocation ? 1 : 0);

        // R2: Force Sensor Based Reward
        float R2 = 0f;
        if (!gripperState.openGripper && forceFingerA.collisionName == "[Target Dynamic] Box - Top" || forceFingerB.collisionName == "[Target Dynamic] Box - Top")
            R2 = (forceFingerA.fNorm + forceFingerB.fNorm) * 0.0005f * speed * (!boxTopInCollisionWithUnpackLocation ? 1 : 0);

        // R3: Box Cover in Designated Unpacking Location
        float distanceBoxTopTargetToUnpackingLocation = Vector3.Distance(boxTargetPosWithNoise, unpackLocationPosWithNoise);
        if (maxDistanceBoxTopTargetToUnpackLocation < 0)
            maxDistanceBoxTopTargetToUnpackLocation = distanceBoxTopTargetToUnpackingLocation;
        distanceBoxTopTargetToUnpackLocationNorm = Mathf.Clamp(distanceBoxTopTargetToUnpackingLocation / maxDistanceBoxTopTargetToUnpackLocation, 0, 1);
        AddReward((1 - distanceBoxTopTargetToUnpackLocationNorm) * 0.001f * speed * (!boxTopInCollisionWithUnpackLocation ? 1 : 0));
        boxTopInCollisionWithUnpackLocation = unpackLocationInCollisionWithBoxTopTarget.inContactWithTarget &&
            unpackLocationInCollisionWithBoxTopTarget.gameObjectInCollision.name == "[Target Dynamic] Box - Top";
        float R3 = (boxTopInCollisionWithUnpackLocation ? 1 : 0) * 0.005f * speed;

        // R4: Distance to Initial Position
        float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
        if (maxDistanceAgentToInitPos < 0)
            maxDistanceAgentToInitPos = distanceAgentToInitPos;
        distanceAgentToInitPosNorm = Mathf.Clamp(distanceAgentToInitPos / maxDistanceAgentToInitPos, 0, 1);
        float R4 = (1 - distanceAgentToInitPosNorm) * 0.0005f * speed * (boxTopInCollisionWithUnpackLocation ? 1 : 0);
        distanceToBase = distanceAgentToInitPos;

        // Implement the logic from your first AgentSpecificReward method
        float pickingAndDroppingReward = R1 + R2 + R3 + R4;

        return pickingAndDroppingReward; // Calculate and return the picking & dropping-specific reward
    }


    // void AgentSpecificReward()
    // {

    //     // R1: Distance to Drawer Handle
    //     float distanceAgentToHandle = Vector3.Distance(agentTR.localPosition, handleTargetPosWithNoise);
    //     if (maxDistanceAgentToHandle < 0)
    //         maxDistanceAgentToHandle = distanceAgentToHandle;
    //     distanceAgentToHandleNorm = Mathf.Clamp(distanceAgentToHandle / maxDistanceAgentToHandle, 0, 1);
    //     AddReward((1 - distanceAgentToHandleNorm) * 0.0005f * speed * (1 - drawerOpenDistanceNorm));

    //     // R2: Drawer Open Distance
    //     float drawerOpenDistance = Vector3.Distance(handleTargetPosWithNoise, sceneManager.drawerHandleInitPos);
    //     drawerOpenDistanceNorm = Mathf.Clamp(drawerOpenDistance / (sceneManager.drawerCJ.linearLimit.limit * 2f), 0, 1);
    //     AddReward(drawerOpenDistanceNorm * 0.005f * speed);

    //     // R3: Force Sensor Based Reward
    //     if (!gripperState.openGripper)
    //         AddReward((forceFingerA.fNorm + forceFingerB.fNorm) * 0.0005f * speed * (1 - drawerOpenDistanceNorm));

    //     // R4: Distance to Initial Position
    //     float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
    //     if (maxDistanceAgentToInitPos < 0)
    //         maxDistanceAgentToInitPos = distanceAgentToInitPos;
    //     distanceAgentToInitPosNorm = Mathf.Clamp(distanceAgentToInitPos / maxDistanceAgentToInitPos, 0, 1);
    //     AddReward((1 - distanceAgentToInitPosNorm) * 0.0005f * speed * drawerOpenDistanceNorm);
    //     distanceToBase = distanceAgentToInitPos;

    //     // Primary Goal is Achieved and Agent is Wihin Initial Position
    //     if (distanceAgentToInitPos < 0.1f && drawerOpenDistanceNorm >= 0.95f && (StepCount >= (maxStep / 10)) && !agentControlled)
    //     {
    //         SetReward(1000f);
    //         PrintEpisodeInfo(true);
    //         EndEpisode();
    //     }
        
    // }

    // For Printing and Debugging Purposes
    void PrintEpisodeInfo(bool invariantStepCount)
    {
        if (currentlyHeuristic && (invariantStepCount || ((StepCount) % 1000) == 0))
        {
            Debug.Log("Cumulative reward: " + GetCumulativeReward()
               + ", for episode: " + episode + ", at environment step: " + StepCount
               + ", with wall-clock time: " + (Time.realtimeSinceStartup - actualTime) + " sec.");
        }
    }

    // Heuristic Actions (For Demonstration / Imitation Purposes)
    public float[] heuristicActions;
    private bool currentlyHeuristic = false;
    //public override float[] Heuristic()
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        /*
        heuristicActions = new float[4];
        heuristicActions[0] = Input.GetAxis("Vertical");
        heuristicActions[1] = Input.GetAxis("Horizontal");
        heuristicActions[2] = Input.GetAxis("Up_Down");
        heuristicActions[3] = Input.GetAxis("OpenCloseGripper");
        */

        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Vertical");
        continuousActionsOut[1] = Input.GetAxis("Horizontal");
        continuousActionsOut[2] = Input.GetAxis("Up_Down");
        continuousActionsOut[3] = Input.GetAxis("OpenCloseGripper");

        if (ButtonIsBeingPressed() && episodeHasResetForButtonPress)
            for (int i = 0; i < heuristicActions.Length; i++)
                heuristicActions[i] = 0f;
        else
            episodeHasResetForButtonPress = false;
        currentlyHeuristic = true;
        //return heuristicActions;
    }

    // Functions to Reset Internal Variables and Scene Parameters
    private bool episodeHasResetForButtonPress;
    bool ButtonIsBeingPressed()
    {
        if (Input.GetButton("Vertical") || Input.GetButton("Horizontal") ||
            Input.GetButton("Up_Down") || Input.GetButton("OpenCloseGripper"))
            return true;
        return false;
    }

    void ResetEpisodeParameters()
    {
        sceneManager.ResetScene();
        maxDistanceAgentToHandle = -1f;
        maxDistanceAgentToInitPos = -1f;
        // +++ 2
        maxDistanceAgentToBoxTopTarget = -1f;
        maxDistanceBoxTopTargetToUnpackLocation = -1f;
        //currentTask = TaskState.Pull;
    }

    // Randomise Next Episode Parameters
    void RandomizeEpisodeVariables()
    {
        sceneManager.RandomizeAgent();
        sceneManager.RandomizeScene_ExpertPull();
        // +++ ? ok just not change at the first time
        sceneManager.RandomizeScene_ExpertPickDrop();
    }

    // Returns Filtered Position via a Low-Pass Filter (Optional)
    Vector3 LowPassFilteredPosition(Vector3 unfilteredPos, int[] parsedIndeces)
    {
        if (!applyFilter)
            return unfilteredPos;

        Vector3 filteredPos = new Vector3(
            ReturnFilteredFloat(unfilteredPos.x, parsedIndeces[0]),
            ReturnFilteredFloat(unfilteredPos.y, parsedIndeces[1]),
            ReturnFilteredFloat(unfilteredPos.z, parsedIndeces[2]));
        return filteredPos;
    }

    float ReturnFilteredFloat(float unfilteredValue, int index)
    {
        lowPassFilter[index].UpdateFilter(unfilteredValue);
        float filteredValue = lowPassFilter[index].filteredValue;
        return filteredValue;
    }
}