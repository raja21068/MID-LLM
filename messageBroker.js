const redis = require('redis');
const subscriber = redis.createClient();
const publisher = redis.createClient();

subscriber.on('message', (channel, message) => {
    console.log(`Received message on channel ${channel}: ${message}`);
    // Further processing can be added here
});

subscriber.subscribe('blockchainChannel');

function publishMessage(channel, message) {
    publisher.publish(channel, message);
}
