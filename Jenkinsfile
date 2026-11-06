// Declarative CI pipeline for stackxai.
// Stages: lint -> unit tests -> build image -> smoke test -> (optional) push -> (optional) deploy.
// Requires an agent with the Docker CLI and access to a Docker daemon (see jenkins/Dockerfile).

pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    // Jenkins on localhost cannot receive GitHub webhooks, so poll the repository.
    triggers {
        pollSCM('H/2 * * * *')
    }

    parameters {
        booleanParam(name: 'PUSH_IMAGE', defaultValue: false,
                     description: 'Push the image to Docker Hub (needs credentials "dockerhub-creds")')
        booleanParam(name: 'DEPLOY_K8S', defaultValue: false,
                     description: 'Run the evaluation as a Kubernetes Job (needs PUSH_IMAGE and credentials "kubeconfig")')
    }

    environment {
        DOCKERHUB_USER = 'melchi0r'   
        IMAGE_NAME     = "${DOCKERHUB_USER}/stackxai"
        TEST_IMAGE     = "stackxai-test:${BUILD_NUMBER}"
        TEST_CONTAINER = "stackxai-tests-${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout info') {
            steps {
                script {
                    env.GIT_SHORT = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.IMAGE_TAG = "${env.BUILD_NUMBER}-${env.GIT_SHORT}"
                }
                sh 'git log -1 --pretty=format:"%h %an: %s"'
                echo "Image tag: ${env.IMAGE_TAG}"
            }
        }

        stage('Build test image') {
            steps {
                sh 'docker build --target test -t "$TEST_IMAGE" .'
            }
        }

        stage('Lint') {
            steps {
                sh 'docker run --rm "$TEST_IMAGE" sh -c "ruff check . && ruff format --check ."'
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    docker rm -f "$TEST_CONTAINER" >/dev/null 2>&1 || true
                    docker run --name "$TEST_CONTAINER" "$TEST_IMAGE" \
                        pytest --cov --cov-report=xml:/tmp/coverage.xml --cov-report=term \
                               --cov-fail-under=90 --junitxml=/tmp/junit.xml
                '''
            }
            post {
                always {
                    sh '''
                        docker cp "$TEST_CONTAINER":/tmp/junit.xml junit.xml || true
                        docker cp "$TEST_CONTAINER":/tmp/coverage.xml coverage.xml || true
                        docker rm -f "$TEST_CONTAINER" >/dev/null 2>&1 || true
                    '''
                    junit testResults: 'junit.xml', allowEmptyResults: true
                    archiveArtifacts artifacts: 'coverage.xml', allowEmptyArchive: true
                }
            }
        }

        stage('Build Docker image') {
            steps {
                sh '''
                    docker build --target runtime \
                        -t "$IMAGE_NAME:$IMAGE_TAG" -t "$IMAGE_NAME:latest" .
                '''
            }
        }

        stage('Smoke test image') {
            steps {
                sh '''
                    docker run --rm "$IMAGE_NAME:$IMAGE_TAG" --version
                    docker run --rm "$IMAGE_NAME:$IMAGE_TAG" evaluate --cv 2 --seed 42
                '''
            }
        }

        stage('Push to Docker Hub') {
            when { expression { params.PUSH_IMAGE } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds',
                                                  usernameVariable: 'DH_USER',
                                                  passwordVariable: 'DH_TOKEN')]) {
                    sh '''
                        echo "$DH_TOKEN" | docker login -u "$DH_USER" --password-stdin
                        docker push "$IMAGE_NAME:$IMAGE_TAG"
                        docker push "$IMAGE_NAME:latest"
                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to Kubernetes (Minikube)') {
            when { expression { params.PUSH_IMAGE && params.DEPLOY_K8S } }
            steps {
                withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                    sh '''
                        kubectl delete job stackxai-evaluate --ignore-not-found
                        sed "s|IMAGE_PLACEHOLDER|$IMAGE_NAME:$IMAGE_TAG|" k8s/job.yaml | kubectl apply -f -
                        kubectl wait --for=condition=complete job/stackxai-evaluate --timeout=300s
                        kubectl logs job/stackxai-evaluate
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker rmi "$TEST_IMAGE" >/dev/null 2>&1 || true'
        }
        success { echo 'Pipeline finished successfully.' }
        failure { echo 'Pipeline failed - see the stage logs above.' }
    }
}
