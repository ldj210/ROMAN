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
    // pull
    public Transform drawerhandleTR;
    // pick and drop
    public Transform boxTopTargetTR;
    public Transform unpackingLocationTR;
    public ObjectInCollisionWith unpackLocationInCollisionWithBoxTopTarget;
    // door open
    public Transform doorHandleTargetTR;
    // pick and place
    public Transform rackTargetTR;
    public Transform rackLocationTargetTR;
    public ObjectInCollisionWith objectInCollisionWith;
    // pick and insert
    public Transform vialTargetTR;
    public ObjectInCollisionWith vialInCollisionWith;
    // push
    public Transform endLocationTargetTR;
    // button
    public Transform buttonTargetTR;


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

    // button
    [System.NonSerialized]
    public ButtonFunction buttonFunction;

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

        //button
        buttonFunction = buttonTargetTR.GetComponent<ButtonFunction>();

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
    // pull
    private Vector3 handleTargetPosWithNoise;
    // pick and drop
    private Vector3 boxTargetPosWithNoise, unpackLocationPosWithNoise;
    // door open
    private Vector3 doorHandleTargetPosWithNoise;
    // pick and place
    private Vector3 rackTargetPosWithNoise, rackLocationTargetPosWithNoise;
    // pick and insert
    private Vector3 vialTargetPosWithNoise;
    // push
    private Vector3 rackEndLocationTargetPosWithNoise;
    // button
    private Vector3 buttonTargetPosWithNoise;

    // State Space
    public override void CollectObservations(VectorSensor sensor)
    {
        // Input: Robot Proprioception (6)
        sensor.AddObservation(agentTR.localPosition);
        sensor.AddObservation(agentRB.velocity);

        // Input: Robot Force Sensor (2)
        sensor.AddObservation(forceFingerA.fNorm);
        sensor.AddObservation(forceFingerB.fNorm);

        // Input: Noisy Environment Data (3) for pull
        handleTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(drawerhandleTR.localPosition, environmentNoise.posNoiseMeters);
        handleTargetPosWithNoise = LowPassFilteredPosition(handleTargetPosWithNoise, new int[] { 0, 1, 2 });
        sensor.AddObservation(handleTargetPosWithNoise);
        
        // Input: Noisy Environment Data (6) for pick and drop
        boxTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(boxTopTargetTR.localPosition, environmentNoise.posNoiseMeters);
        boxTargetPosWithNoise = LowPassFilteredPosition(boxTargetPosWithNoise, new int[] { 3, 4, 5 });
        sensor.AddObservation(boxTargetPosWithNoise);
        unpackLocationPosWithNoise = GaussianDistribution.PositionWGaussianNoise(unpackingLocationTR.localPosition, environmentNoise.posNoiseMeters);
        unpackLocationPosWithNoise = LowPassFilteredPosition(unpackLocationPosWithNoise, new int[] { 6, 7, 8 });
        sensor.AddObservation(unpackLocationPosWithNoise);
        
        // Input: Noisy Environment Data (3) for door open
        doorHandleTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(doorHandleTargetTR.localPosition, environmentNoise.posNoiseMeters);
        doorHandleTargetPosWithNoise = LowPassFilteredPosition(doorHandleTargetPosWithNoise, new int[] { 9, 10, 11 });
        sensor.AddObservation(doorHandleTargetPosWithNoise);

        // Input: Noisy Environment Data (6) for pickplace (push)
        rackTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(rackTargetTR.localPosition, environmentNoise.posNoiseMeters);
        rackTargetPosWithNoise = LowPassFilteredPosition(rackTargetPosWithNoise, new int[] { 12, 13, 14 });
        sensor.AddObservation(rackTargetPosWithNoise);
        rackLocationTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(rackLocationTargetTR.localPosition, environmentNoise.posNoiseMeters);
        rackLocationTargetPosWithNoise = LowPassFilteredPosition(rackLocationTargetPosWithNoise, new int[] { 21, 22, 23 });
        sensor.AddObservation(rackLocationTargetPosWithNoise);

        // Input: Noisy Environment Data (6) for pickinsert
        vialTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(vialTargetTR.localPosition, environmentNoise.posNoiseMeters);
        vialTargetPosWithNoise = LowPassFilteredPosition(vialTargetPosWithNoise, new int[] { 15, 16, 17 });
        sensor.AddObservation(vialTargetPosWithNoise);

        // Input: Noisy Environment Data (6) for push
        rackEndLocationTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(endLocationTargetTR.localPosition, environmentNoise.posNoiseMeters);
        rackEndLocationTargetPosWithNoise = LowPassFilteredPosition(rackEndLocationTargetPosWithNoise, new int[] { 21, 22, 23 });
        sensor.AddObservation(rackEndLocationTargetPosWithNoise);

        // Input: Noisy Environment Data (3) for buttion
        buttonTargetPosWithNoise = GaussianDistribution.PositionWGaussianNoise(buttonTargetTR.localPosition, environmentNoise.posNoiseMeters);
        buttonTargetPosWithNoise = LowPassFilteredPosition(buttonTargetPosWithNoise, new int[] { 18, 19, 20 });
        sensor.AddObservation(buttonTargetPosWithNoise);
        sensor.AddObservation(buttonFunction.buttonActivated);

        // common for each Expert
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

        // Apply Agent-Specific Rewards
        if (StepCount >= (MaxStep / (MaxStep <= 80000 ? 100 : stepSafetySync)))
            AgentSpecificReward();

        // Terminate if Agent Divergence is Higher than Threshold
        if (Vector3.Distance(agentDefaultInitPos, agentTR.localPosition) > maxAllowableDivergence)
            EndEpisode();
        
        totalReward = GetCumulativeReward();
        PrintEpisodeInfo(false);
    }

    // Task State Shift
    enum TaskState { Pull, PickDrop, DoorOpen, PickPlace, PickInsert, Push, Button };
    TaskState currentTask = TaskState.Pull;

    void UpdateTaskState() {
        switch (currentTask) {
            case TaskState.Pull:
                Debug.Log("Task =====================================>>>>> Pull");
                if (drawerOpenDistanceNorm >= 0.95f) {// Drawer fully opened
                    currentTask = TaskState.PickDrop;
                    Debug.Log("case TaskState.Pull ===================================== Task-PickDrop:" + currentTask);
                }
                break;

            case TaskState.PickDrop:
                if (drawerOpenDistanceNorm < 0.90f) {// Drawer not fully opened
                    currentTask = TaskState.Pull;
                    Debug.Log("case TaskState.PickDrop ===================================== Task-Pull:" + currentTask);
                }
                else if (boxTopInCollisionWithUnpackLocation)
                {
                    currentTask = TaskState.DoorOpen;
                    Debug.Log("case TaskState.PickDrop ===================================== Task-DoorOpen:" + currentTask);
                }
                // Criteria to switch to dropping, e.g., object is picked
                break;

            case TaskState.DoorOpen:
                if (drawerOpenDistanceNorm < 0.90f) {
                    currentTask = TaskState.Pull;
                    Debug.Log("case TaskState.DoorOpen ===================================== Task-Pull:" + currentTask);
                }
                else if (boxTopInCollisionWithUnpackLocation == false)
                {
                    currentTask = TaskState.PickDrop;
                    Debug.Log("case TaskState.DoorOpen ===================================== Task-PickDrop:" + currentTask);
                }
                else if (doorHingeOpenAngleNorm >= 0.85f) {
                    currentTask = TaskState.PickPlace;
                    Debug.Log("case TaskState.DoorOpen ===================================== Task-PickPlace:" + currentTask);
                }
                // Criteria to switch to dropping, e.g., object is picked
                break;

            case TaskState.PickPlace:
                if (drawerOpenDistanceNorm < 0.90f) {
                    currentTask = TaskState.Pull;
                    Debug.Log("case TaskState.PickPlace ===================================== Task-Pull:" + currentTask);
                }
                else if (boxTopInCollisionWithUnpackLocation == false)
                {
                    currentTask = TaskState.PickDrop;
                    Debug.Log("case TaskState.PickPlace ===================================== Task-PickDrop:" + currentTask);
                }
                else if (doorHingeOpenAngleNorm < 0.85f) {
                    currentTask = TaskState.DoorOpen;
                    Debug.Log("case TaskState.PickPlace ===================================== Task-DoorOpen:" + currentTask);
                }
                else if (distanceRackToEndLocationNorm <= 0.05f)
                {
                    currentTask = TaskState.Push;
                    Debug.Log("case TaskState.PickPlace ===================================== Task-Push:" + currentTask);
                }
                else if (rackInContactWithRack)
                {
                    currentTask = TaskState.PickInsert;
                    Debug.Log("case TaskState.PickPlace ===================================== Task-PickInsert:" + currentTask);
                }
                // Criteria to switch to dropping, e.g., object is picked
                break;

            case TaskState.PickInsert:
                if (drawerOpenDistanceNorm < 0.90f) {
                    currentTask = TaskState.Pull;
                    Debug.Log("case TaskState.PickInsert ===================================== Task-Pull:" + currentTask);
                }
                else if (boxTopInCollisionWithUnpackLocation == false)
                {
                    currentTask = TaskState.PickDrop;
                    Debug.Log("case TaskState.PickInsert ===================================== Task-PickDrop:" + currentTask);
                }
                else if (doorHingeOpenAngleNorm < 0.85f) {
                    currentTask = TaskState.DoorOpen;
                    Debug.Log("case TaskState.PickInsert ===================================== Task-DoorOpen:" + currentTask);
                }
                else if (rackInContactWithRack == false)
                {
                    currentTask = TaskState.PickPlace;
                    Debug.Log("case TaskState.PickInsert ===================================== Task-PickPlace:" + currentTask);
                }
                else if (vialTargetInCollsionWithRack)
                {
                    currentTask = TaskState.Push;
                    Debug.Log("case TaskState.PickInsert ===================================== Task-Push:" + currentTask);
                }
                // Criteria to switch to dropping, e.g., object is picked
                break;

            case TaskState.Push:
                if (drawerOpenDistanceNorm < 0.90f) {
                    currentTask = TaskState.Pull;
                    Debug.Log("case TaskState.Push ===================================== Task-Pull:" + currentTask);
                }
                else if (boxTopInCollisionWithUnpackLocation == false)
                {
                    currentTask = TaskState.PickDrop;
                    Debug.Log("case TaskState.Push ===================================== Task-PickDrop:" + currentTask);
                }
                else if (doorHingeOpenAngleNorm < 0.85f) {
                    currentTask = TaskState.DoorOpen;
                    Debug.Log("case TaskState.Push ===================================== Task-DoorOpen:" + currentTask);
                }
                // else if (rackInContactWithRack == false)
                // {
                //     currentTask = TaskState.PickPlace;
                //     Debug.Log("Task-PickPlace:" + currentTask);
                // }
                else if (vialTargetInCollsionWithRack == false)
                {
                    currentTask = TaskState.PickInsert;
                    Debug.Log("case TaskState.Push ===================================== Task-PickInsert:" + currentTask);
                }
                else if (distanceRackToEndLocationNorm <= 0.05f)
                {
                    currentTask = TaskState.Button;
                    Debug.Log("case TaskState.Push ===================================== Task-Button:" + currentTask);
                }
                // Criteria to switch to dropping, e.g., object is picked
                break;

            case TaskState.Button:
                if (drawerOpenDistanceNorm < 0.90f) {
                    currentTask = TaskState.Pull;
                    Debug.Log("case TaskState.Button ===================================== Task-Pull:" + currentTask);
                }
                else if (boxTopInCollisionWithUnpackLocation == false)
                {
                    currentTask = TaskState.PickDrop;
                    Debug.Log("case TaskState.Button ===================================== Task-PickDrop:" + currentTask);
                }
                else if (doorHingeOpenAngleNorm < 0.85f) {
                    currentTask = TaskState.DoorOpen;
                    Debug.Log("case TaskState.Button ===================================== Task-DoorOpen:" + currentTask);
                }
                // else if (rackInContactWithRack == false)
                // {
                //     currentTask = TaskState.PickPlace;
                //     Debug.Log("Task-PickPlace:" + currentTask);
                // }
                else if (vialTargetInCollsionWithRack == false)
                {
                    currentTask = TaskState.PickInsert;
                    Debug.Log("case TaskState.Button ===================================== Task-PickInsert:" + currentTask);
                }
                // else if (distanceRackToEndLocationNorm > 0.05f)
                // {
                //     currentTask = TaskState.Push;
                //     Debug.Log("case TaskState.Button ===================================== Task-Push:" + currentTask);
                // }
                else if (buttonFunction.buttonActivated)
                {
                    currentTask = TaskState.Pull;
                    Debug.Log("case TaskState.Button ===================================== Task-Pull:" + currentTask);
                }
                // Criteria to switch to dropping, e.g., object is picked
                break;
        }
    }

    // Agent Specific Rewards
    public float totalReward = 0;
    public float distanceAgentToInitPosNorm;
    public float maxDistanceAgentToInitPos = -1f;
    // pull
    public float distanceAgentToHandleNorm, drawerOpenDistanceNorm;
    public float maxDistanceAgentToHandle = -1f;
    // pickanddrop
    public bool boxTopInCollisionWithUnpackLocation = false;
    public float distanceAgentToBoxTopTargetNorm, distanceBoxTopTargetToUnpackLocationNorm;
    public float maxDistanceAgentToBoxTopTarget = -1f, maxDistanceBoxTopTargetToUnpackLocation = -1f;
    // dooropen
    public float distanceAgentToDoorHandleNorm, doorHingeOpenAngleNorm;
    public float maxDistanceAgentToDoorHandle = -1f;
    // pickplace
    public bool rackInContactWithRack = false;
    public float distanceAgentToRackNorm, distanceRackToRackLocationNorm;
    public float maxDistanceAgentToRack = -1f, maxDistanceRackToRackLocation = -1f;
    // pickinsert
    public bool vialTargetInCollsionWithRack = false;
    public float distanceAgentToVialTargetNorm, distanceVialTargetToRackTargetNorm, vialRackHeightDiffNorm;
    public float maxDistanceAgentToVialTarget = -1f, maxDistanceVialTargetToRackTarget = -1f;
    public float agentRackHeightDiff, actualVialRackHeightDiff;
    public bool vialHeightPenalizedThisEpisode = false;
    private const float maxVialRackHeight = 0.4f;
    private const float vialAgentDisplacement = 0.275f;
    // push
    public float distanceRackToEndLocation;
    public float distanceRackToEndLocationNorm;
    public float maxDistanceRackToEndLocation = -1f;
    // button
    public float distanceAgentToButtonNorm;
    public float maxDistanceAgentToButton = -1f;
    
    [System.NonSerialized]
    public float distanceToBase = 0.0f;

    // +++
    void AgentSpecificReward()
    {
        // pull
        float drawerOpenDistance = Vector3.Distance(handleTargetPosWithNoise, sceneManager.drawerHandleInitPos);
        drawerOpenDistanceNorm = Mathf.Clamp(drawerOpenDistance / (sceneManager.drawerCJ.linearLimit.limit * 2f), 0, 1);

        // pickdrop
        boxTopInCollisionWithUnpackLocation = unpackLocationInCollisionWithBoxTopTarget.inContactWithTarget &&
            unpackLocationInCollisionWithBoxTopTarget.gameObjectInCollision.name == "[Target Dynamic] Box - Top";

        // dooropen
        float maxDoorHingeOpenAngleLimit = sceneManager.doorHingeCJ.highAngularXLimit.limit;
        float currentDoorHingeOpenAngle = 
            Mathf.Clamp(sceneManager.doorHingeTR.localEulerAngles.y > 180.0f ?
                Mathf.Abs(270 - sceneManager.doorHingeTR.localEulerAngles.y) :
                (90 + sceneManager.doorHingeTR.localEulerAngles.y)
                , 0f, 120f);
        doorHingeOpenAngleNorm = Mathf.Clamp(currentDoorHingeOpenAngle / maxDoorHingeOpenAngleLimit, 0, 1);

        // pickplace
        rackInContactWithRack = objectInCollisionWith.inContactWithTarget &&
            objectInCollisionWith.gameObjectInCollision.name == "[Target Static] Rack Location";

        // pickinsert 
        vialTargetInCollsionWithRack = vialInCollisionWith.inContactWithTarget &&
            vialInCollisionWith.gameObjectInCollision.name == "Bottom Goal" &&
            vialInCollisionWith.gameObjectInCollision.transform.parent.name == "[Target Dynamic] Rack";
        
        // push
        distanceRackToEndLocation = Mathf.Abs(rackTargetPosWithNoise.z - rackEndLocationTargetPosWithNoise.z);
        if (maxDistanceRackToEndLocation < 0)
            maxDistanceRackToEndLocation = distanceRackToEndLocation;
        distanceRackToEndLocationNorm = Mathf.Clamp(distanceRackToEndLocation / maxDistanceRackToEndLocation, 0, 1);

        // Debug.Log("drawerOpenDistanceNorm>=0.95?:" + drawerOpenDistanceNorm);
        // Debug.Log("boxTopInCollisionWithUnpackLocation == true?:" + boxTopInCollisionWithUnpackLocation);
        // Debug.Log("doorHingeOpenAngleNorm >= 0.85f?:" + doorHingeOpenAngleNorm);
        // Debug.Log("rackInContactWithRack == true?:" + rackInContactWithRack);
        // Debug.Log("vialTargetInCollsionWithRack == true?:" + vialTargetInCollsionWithRack);
        // Debug.Log("distanceRackToEndLocationNorm <= 0.05f?:" + distanceRackToEndLocationNorm);
        // Debug.Log("buttonFunction.buttonActivated == true?:" + buttonFunction.buttonActivated);
        UpdateTaskState();
        // Debug.Log("real-time-currentTask:" + currentTask);

        if (currentTask == TaskState.Pull)
        {
            AddReward(CalculatePullingReward());
        }
        else if (currentTask == TaskState.PickDrop)
        {
            AddReward(CalculatePickingAndDroppingReward());
        }
        else if (currentTask == TaskState.DoorOpen)
        {
            AddReward(CalculateDoorOpenningReward());
        }
        else if (currentTask == TaskState.PickPlace)
        {
            AddReward(CalculatePickPlacingReward());
        }
        else if (currentTask == TaskState.PickInsert)
        {
            AddReward(CalculatePickInsertingReward());
        }
        else if (currentTask == TaskState.Push)
        {
            AddReward(CalculatePushingReward());
        }
        else if (currentTask == TaskState.Button)
        {
            AddReward(CalculateButtonReward());
        }

        // The next line can be for all expert reward 1 for 7
        float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
        // Primary Goal is Achieved and Agent is Wihin Initial Position
        // if (distanceAgentToInitPos < 0.1f && drawerOpenDistanceNorm >= 0.95f && boxTopInCollisionWithUnpackLocation && doorHingeOpenAngleNorm >= 0.85f && rackInContactWithRack && vialTargetInCollsionWithRack && distanceRackToEndLocationNorm <= 0.05f && buttonFunction.buttonActivated && (StepCount >= (MaxStep / 10)) && !agentControlled)
        if (vialTargetInCollsionWithRack &&
            distanceRackToEndLocationNorm <= 0.15f &&
            buttonFunction.buttonActivated &&
            distanceAgentToInitPos <= 0.3f
            && (StepCount >= (MaxStep / 100)))
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

    float CalculateDoorOpenningReward()
    {
        // R1: Distance to Cabinet Door Handle
        float distanceAgentToHandle = Vector3.Distance(agentTR.localPosition, doorHandleTargetPosWithNoise);
        if (maxDistanceAgentToDoorHandle < 0)
            maxDistanceAgentToDoorHandle = distanceAgentToHandle;
        distanceAgentToDoorHandleNorm = Mathf.Clamp(distanceAgentToHandle / maxDistanceAgentToDoorHandle, 0, 1);
        float R1 = (1 - distanceAgentToDoorHandleNorm) * 0.0005f * speed * (1f - doorHingeOpenAngleNorm);

        // R2: Door Angle Open Distance
        float maxDoorHingeOpenAngleLimit = sceneManager.doorHingeCJ.highAngularXLimit.limit;
        float currentDoorHingeOpenAngle = 
            Mathf.Clamp(sceneManager.doorHingeTR.localEulerAngles.y > 180.0f ?
                Mathf.Abs(270 - sceneManager.doorHingeTR.localEulerAngles.y) :
                (90 + sceneManager.doorHingeTR.localEulerAngles.y)
                , 0f, 120f);
        doorHingeOpenAngleNorm = Mathf.Clamp(currentDoorHingeOpenAngle / maxDoorHingeOpenAngleLimit, 0, 1);
        float R2 = doorHingeOpenAngleNorm * 0.005f * speed;

        // R3: Force Sensor Based Reward
        float R3 = (forceFingerA.fNorm + forceFingerB.fNorm) * 0.0005f * speed * (1f - doorHingeOpenAngleNorm);

        // R4: Distance to Initial Position
        float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
        if (maxDistanceAgentToInitPos < 0)
            maxDistanceAgentToInitPos = distanceAgentToInitPos;
        distanceAgentToInitPosNorm = Mathf.Clamp(distanceAgentToInitPos / maxDistanceAgentToInitPos, 0, 1);
        float R4 = (1 - distanceAgentToInitPosNorm) * 0.0005f * speed * doorHingeOpenAngleNorm;
        distanceToBase = distanceAgentToInitPos;


        // Implement the logic from your first AgentSpecificReward method
        float DoorAndOpenningReward = R1 + R2 + R3 + R4;

        return DoorAndOpenningReward;
    }

    float CalculatePickPlacingReward()
    {
        // R1: Distance to Rack
        float distanceAgentToRack = Vector3.Distance(agentTR.localPosition, rackTargetPosWithNoise);
        if (maxDistanceAgentToRack < 0)
            maxDistanceAgentToRack = distanceAgentToRack;
        distanceAgentToRackNorm = Mathf.Clamp(distanceAgentToRack / maxDistanceAgentToRack, 0, 1);
        float R1 = (1 - distanceAgentToRackNorm) * 0.0005f * speed * (!rackInContactWithRack ? 1 : 0);

        // R2: Force Sensor Based Reward
        float R2 = 0f;
        if (!gripperState.openGripper)
            R2 = (forceFingerA.fNorm + forceFingerB.fNorm) * 0.0005f * (!rackInContactWithRack ? 1 : 0);

        // R3: Rack in Designated Rack Target Location
        float distanceRackToRackLocation = Vector3.Distance(rackTargetPosWithNoise, rackLocationTargetPosWithNoise);
        if (maxDistanceRackToRackLocation < 0)
            maxDistanceRackToRackLocation = distanceRackToRackLocation;
        distanceRackToRackLocationNorm = Mathf.Clamp(distanceRackToRackLocation / maxDistanceRackToRackLocation, 0, 1);
        float R3_1 = (1 - distanceRackToRackLocationNorm) * 0.001f * speed;

        rackInContactWithRack = objectInCollisionWith.inContactWithTarget &&
            objectInCollisionWith.gameObjectInCollision.name == "[Target Static] Rack Location";
        float R3_2 = (rackInContactWithRack ? 1 : 0) * 0.005f * speed;

        // R4: Distance to Initial Position
        float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
        if (maxDistanceAgentToInitPos < 0)
            maxDistanceAgentToInitPos = distanceAgentToInitPos;
        distanceAgentToInitPosNorm = Mathf.Clamp(distanceAgentToInitPos / maxDistanceAgentToInitPos, 0, 1);
        float R4 = (1 - distanceAgentToInitPosNorm) * 0.0005f * speed * (rackInContactWithRack ? 1 : 0);
        distanceToBase = distanceAgentToInitPos;

        // Implement the logic from your first AgentSpecificReward method
        float PickAndPlacingReward = R1 + R2 + R3_1 + R3_2 + R4;

        return PickAndPlacingReward;
    }

    float CalculatePickInsertingReward()
    {
        // R1: Distance to Vial Target
        float distanceAgentToRack = Vector3.Distance(agentTR.localPosition, vialTargetPosWithNoise);
        if (maxDistanceAgentToVialTarget < 0)
            maxDistanceAgentToVialTarget = distanceAgentToRack;
        distanceAgentToVialTargetNorm = Mathf.Clamp(distanceAgentToRack / maxDistanceAgentToVialTarget, 0, 1);
        float R1 = (1 - distanceAgentToVialTargetNorm) * 0.0005f * speed * (!vialTargetInCollsionWithRack ? 1 : 0);

        // R2: Force Sensor Based Reward
        float R2 = 0f;
        if (!gripperState.openGripper)
            R2 = (forceFingerA.fNorm + forceFingerB.fNorm) * 0.0005f * speed * (!vialTargetInCollsionWithRack ? 1 : 0);

        // R3: Distance Vial Target to Rack
        float distanceRackToRackLocation = Vector3.Distance(vialTargetPosWithNoise, rackTargetPosWithNoise);
        if (maxDistanceVialTargetToRackTarget < 0)
            maxDistanceVialTargetToRackTarget = distanceRackToRackLocation;
        distanceVialTargetToRackTargetNorm = Mathf.Clamp(distanceRackToRackLocation / maxDistanceVialTargetToRackTarget, 0, 1);
        float R3 = (1 - distanceVialTargetToRackTargetNorm) * 0.001f * speed * (!vialTargetInCollsionWithRack ? 1 : 0);

        // R4: Vial Target Inside Rack
        vialTargetInCollsionWithRack = vialInCollisionWith.inContactWithTarget &&
            vialInCollisionWith.gameObjectInCollision.name == "Bottom Goal" &&
            vialInCollisionWith.gameObjectInCollision.transform.parent.name == "[Target Dynamic] Rack";
        float R4 = (vialTargetInCollsionWithRack ? 1 : 0) * 0.005f * speed;

        agentRackHeightDiff = Mathf.Clamp(agentTR.localPosition.y - rackTargetPosWithNoise.y, 0, 1);
        actualVialRackHeightDiff = Mathf.Clamp(agentRackHeightDiff - vialAgentDisplacement, 0, 1);
        vialRackHeightDiffNorm = Mathf.Clamp(actualVialRackHeightDiff / maxVialRackHeight, 0, 1);

        // R5: Height Penalisation
        float R5 = 0f;
        if (vialTargetInCollsionWithRack && !vialHeightPenalizedThisEpisode)
        {
            vialHeightPenalizedThisEpisode = true;
            agentRackHeightDiff = Mathf.Clamp(agentTR.localPosition.y - rackTargetPosWithNoise.y, 0, 1);
            actualVialRackHeightDiff = Mathf.Clamp(agentRackHeightDiff - vialAgentDisplacement, 0, 1);
            vialRackHeightDiffNorm = Mathf.Clamp(actualVialRackHeightDiff / maxVialRackHeight, 0, 1);
            R5 = -vialRackHeightDiffNorm * 1000f;
        }

        // R6: Distance to Initial Position
        float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
        if (maxDistanceAgentToInitPos < 0)
            maxDistanceAgentToInitPos = distanceAgentToInitPos;
        distanceAgentToInitPosNorm = Mathf.Clamp(distanceAgentToInitPos / maxDistanceAgentToInitPos, 0, 1);
        float R6 = (1 - distanceAgentToInitPosNorm) * 0.0005f * speed * (vialTargetInCollsionWithRack ? 1 : 0);
        distanceToBase = distanceAgentToInitPos;

        // Implement the logic from your first AgentSpecificReward method
        float PickAndPlacingReward = R1 + R2 + R3 + R4 + R5 + R6;

        return PickAndPlacingReward;
    }

    float CalculatePushingReward()
    {
        // R1: Distance to Rack
        float distanceAgentToRack = Vector3.Distance(agentTR.localPosition, rackTargetPosWithNoise);
        if (maxDistanceAgentToRack < 0)
            maxDistanceAgentToRack = distanceAgentToRack;
        distanceAgentToRackNorm = Mathf.Clamp(distanceAgentToRack / maxDistanceAgentToRack, 0, 1);
        float R1 = (1 - distanceAgentToRackNorm) * 0.0005f * speed * (distanceRackToEndLocationNorm);

        // R2: Distance Rack to Conveyor Belt Position
        distanceRackToEndLocation = Mathf.Abs(rackTargetPosWithNoise.z - rackEndLocationTargetPosWithNoise.z);
        if (maxDistanceRackToEndLocation < 0)
            maxDistanceRackToEndLocation = distanceRackToEndLocation;
        distanceRackToEndLocationNorm = Mathf.Clamp(distanceRackToEndLocation / maxDistanceRackToEndLocation, 0, 1);
        float R2 = (1 - distanceRackToEndLocationNorm) * 0.005f * speed;

        // R3: Distance to Initial Position
        float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
        if (maxDistanceAgentToInitPos < 0)
            maxDistanceAgentToInitPos = distanceAgentToInitPos;
        distanceAgentToInitPosNorm = Mathf.Clamp(distanceAgentToInitPos / maxDistanceAgentToInitPos, 0, 1);
        float R3 = (1 - distanceAgentToInitPosNorm) * 0.0005f * speed * (1f - distanceRackToEndLocationNorm);
        distanceToBase = distanceAgentToInitPos;

        // Implement the logic from your first AgentSpecificReward method
        float PushingReward = R1 + R2 + R3;

        return PushingReward;
    }

    float CalculateButtonReward()
    {
        // R1: Distance to Button
        float distanceAgentToButton = Vector3.Distance(agentTR.localPosition, buttonTargetPosWithNoise);
        if (maxDistanceAgentToButton < 0)
            maxDistanceAgentToButton = distanceAgentToButton;
        distanceAgentToButtonNorm = Mathf.Clamp(distanceAgentToButton / maxDistanceAgentToButton, 0, 1);
        float R1 = (1 - distanceAgentToButtonNorm) * 0.0005f * speed * (buttonFunction.buttonActivated ? 0 : 1);

        // R2: Button Pressed and Activated (Binary)
        float R2 = buttonFunction.buttonActivated ? 0.005f * speed : 0f;

        // R3: Distance to Initial Position
        float distanceAgentToInitPos = Vector3.Distance(agentTR.localPosition, agentDefaultInitPos);
        if (maxDistanceAgentToInitPos < 0)
            maxDistanceAgentToInitPos = distanceAgentToInitPos;
        distanceAgentToInitPosNorm = Mathf.Clamp(distanceAgentToInitPos / maxDistanceAgentToInitPos, 0, 1);
        float R3 = (1 - distanceAgentToInitPosNorm) * 0.0005f * speed * (buttonFunction.buttonActivated ? 1 : 0);
        distanceToBase = distanceAgentToInitPos;

        // Implement the logic from your first AgentSpecificReward method
        float ButtonReward = R1 + R2 + R3;

        return ButtonReward;


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
            // Debug.Log("Cumulative reward: " + GetCumulativeReward()
            //    + ", for episode: " + episode + ", at environment step: " + StepCount
            //    + ", with wall-clock time: " + (Time.realtimeSinceStartup - actualTime) + " sec.");
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
        maxDistanceAgentToInitPos = -1f;
        // pull
        maxDistanceAgentToHandle = -1f;
        // pickdrop;
        maxDistanceAgentToBoxTopTarget = -1f;
        maxDistanceBoxTopTargetToUnpackLocation = -1f;
        // dooropen
        maxDistanceAgentToDoorHandle = -1f;
        // pickplace
        maxDistanceAgentToRack = -1f;
        maxDistanceRackToRackLocation = -1f;
        // pickinsert
        vialHeightPenalizedThisEpisode = false;
        maxDistanceAgentToVialTarget = -1f;
        maxDistanceVialTargetToRackTarget = -1f;
        // push
        maxDistanceRackToEndLocation = -1f;
        // button
        maxDistanceAgentToButton = -1f;
        buttonFunction.buttonActivated = false;
    }

    // Randomise Next Episode Parameters
    void RandomizeEpisodeVariables()
    {
        sceneManager.RandomizeAgent();
        // pull
        sceneManager.RandomizeScene_ExpertPull();
        // //pickdrop
        // sceneManager.RandomizeScene_ExpertPickDrop();
        // // dooropen
        // sceneManager.RandomizeScene_ExpertRotateOpen();
        // // pickplace
        // sceneManager.RandomizeScene_ExpertPickPlace();
        // // pickinsert
        // sceneManager.RandomizeScene_ExpertPickInsert();
        // // push
        // sceneManager.RandomizeScene_ExpertPush();
        // // button
        // sceneManager.RandomizeScene_ExpertButton();
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